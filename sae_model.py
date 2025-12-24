"""
SAE Model Handler for Flask App

Handles loading Gemma model and GemmaScope SAE, running inference,
extracting activations, and steering generation.
"""

import os
import torch
import torch.nn as nn
import numpy as np
import requests
from functools import partial
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Neuronpedia configuration
NEURONPEDIA_API_KEY = os.getenv("NEURONPEDIA_API_KEY")
NEURONPEDIA_BASE_URL = "https://www.neuronpedia.org/api"

# Cache for Neuronpedia data to avoid repeated API calls
_neuronpedia_cache: dict[str, dict] = {}

# Cache for residual activations to avoid re-running model
_residual_cache: dict[str, dict] = {}


# =============================================================================
# Configuration
# =============================================================================

# Model path - adjust to your local path or use HuggingFace ID
#MODEL_PATH = "D:\\huggingface\\gemma-3-12b-it-null-space-abliterated"  # Local path

# MODEL_PATH = "google/gemma-3-4b-it"  # HuggingFace (requires auth)
MODEL_PATH = "D:\\models\\gemma-3-4b-it"

# SAE Configuration
SAE_REPO = "google/gemma-scope-2-4b-it"
SAE_WIDTH = "262k"
SAE_L0 = "small"

# Available SAE layers per model
SAE_LAYERS_BY_MODEL = {
    "4b": [9, 17, 22, 29],
    "12b": [12, 24, 31, 41],
}

# Neuronpedia only has data for certain layers
NEURONPEDIA_LAYERS_BY_MODEL = {
    "4b": [9, 17],
    "12b": [12],
}

# Auto-detect model size from repo name
def get_available_layers(repo: str) -> list[int]:
    """Get available SAE layers based on repo name."""
    if "12b" in repo.lower():
        return SAE_LAYERS_BY_MODEL["12b"]
    elif "4b" in repo.lower():
        return SAE_LAYERS_BY_MODEL["4b"]
    # Default to 12b layers
    return SAE_LAYERS_BY_MODEL["12b"]


def get_neuronpedia_layers(repo: str) -> list[int]:
    """Get layers that have Neuronpedia data available."""
    if "12b" in repo.lower():
        return NEURONPEDIA_LAYERS_BY_MODEL["12b"]
    elif "4b" in repo.lower():
        return NEURONPEDIA_LAYERS_BY_MODEL["4b"]
    return NEURONPEDIA_LAYERS_BY_MODEL["12b"]


SAE_LAYERS = get_available_layers(SAE_REPO)
NEURONPEDIA_LAYERS = get_neuronpedia_layers(SAE_REPO)


# =============================================================================
# JumpReLU SAE Implementation
# =============================================================================

class JumpReLUSAE(nn.Module):
    """
    JumpReLU Sparse Autoencoder.

    Adapted from Google DeepMind's Gemma Scope tutorial.
    """

    def __init__(self, d_in: int, d_sae: int):
        super().__init__()
        self.w_enc = nn.Parameter(torch.zeros(d_in, d_sae))
        self.w_dec = nn.Parameter(torch.zeros(d_sae, d_in))
        self.threshold = nn.Parameter(torch.zeros(d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_in))

    def encode(self, input_acts: torch.Tensor) -> torch.Tensor:
        """Encode input activations to sparse latent space."""
        pre_acts = input_acts @ self.w_enc + self.b_enc
        mask = (pre_acts > self.threshold)
        acts = mask * torch.nn.functional.relu(pre_acts)
        return acts

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        """Decode sparse latents back to activation space."""
        return acts @ self.w_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass: encode then decode."""
        acts = self.encode(x)
        recon = self.decode(acts)
        return recon


# =============================================================================
# Model Manager
# =============================================================================

class SAEModelManager:
    """
    Manages the Gemma model and SAE for the Flask app.

    Handles lazy loading to avoid loading models on import.
    """

    def __init__(
        self,
        model_path: str = MODEL_PATH,
        sae_layers: list[int] = SAE_LAYERS,
        sae_width: str = SAE_WIDTH,
        sae_l0: str = SAE_L0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model_path = model_path
        self.sae_layers = sae_layers
        self.sae_width = sae_width
        self.sae_l0 = sae_l0
        self.device = device

        # Lazy-loaded components
        self._model = None
        self._tokenizer = None
        self._saes: dict[int, JumpReLUSAE] = {}  # layer -> SAE
        self._layers = None  # Will be set after model loads

        # Disable gradients globally for inference
        torch.set_grad_enabled(False)

    def _find_layers(self, model):
        """
        Find the transformer layers in the model.

        Handles different Gemma architectures:
        - Standard: model.model.layers
        - Gemma 3 multimodal: model.model.language_model.model.layers
        - Gemma 3 text: model.model.text_model.layers
        """
        # Try Gemma 3 text model path (Gemma3ForCausalLM)
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            print("Found layers at: model.model.layers")
            return model.model.layers

        # Try Gemma 3 conditional generation with language_model
        if (hasattr(model, 'language_model') and
            hasattr(model.language_model, 'model') and
            hasattr(model.language_model.model, 'layers')):
            print("Found layers at: model.language_model.model.layers")
            return model.language_model.model.layers

        # Try Gemma 3 multimodal nested path
        if (hasattr(model, 'model') and
            hasattr(model.model, 'language_model') and
            hasattr(model.model.language_model, 'model') and
            hasattr(model.model.language_model.model, 'layers')):
            print("Found layers at: model.model.language_model.model.layers")
            return model.model.language_model.model.layers

        # Try text_model path (some Gemma variants)
        if (hasattr(model, 'model') and
            hasattr(model.model, 'text_model') and
            hasattr(model.model.text_model, 'layers')):
            print("Found layers at: model.model.text_model.layers")
            return model.model.text_model.layers

        # Try direct text_model
        if hasattr(model, 'text_model') and hasattr(model.text_model, 'layers'):
            print("Found layers at: model.text_model.layers")
            return model.text_model.layers

        # Debug: print full model structure to find layers
        print("Could not find layers automatically. Searching model structure...")
        print(f"  model type: {type(model).__name__}")

        def find_layers_recursive(obj, path="model", depth=0):
            if depth > 4:
                return None
            if hasattr(obj, 'layers') and hasattr(obj.layers, '__len__'):
                try:
                    if len(obj.layers) > 0:
                        print(f"  FOUND layers at: {path}.layers (count: {len(obj.layers)})")
                        return obj.layers
                except:
                    pass
            for attr_name in ['model', 'language_model', 'text_model', 'decoder', 'encoder']:
                if hasattr(obj, attr_name):
                    result = find_layers_recursive(
                        getattr(obj, attr_name),
                        f"{path}.{attr_name}",
                        depth + 1
                    )
                    if result is not None:
                        return result
            return None

        layers = find_layers_recursive(model)
        if layers is not None:
            return layers

        # Last resort: print all attributes to help debug
        print("\nModel structure exploration:")
        if hasattr(model, 'model'):
            print(f"  model.model attrs: {[a for a in dir(model.model) if not a.startswith('_')]}")
        if hasattr(model, 'language_model'):
            print(f"  model.language_model attrs: {[a for a in dir(model.language_model) if not a.startswith('_')]}")

        raise AttributeError(
            f"Cannot find transformer layers in model of type {type(model).__name__}. "
            "Please inspect the model structure and update _find_layers()."
        )

    @property
    def layers(self):
        """Get the transformer layers (lazy-loaded with model)."""
        if self._layers is None:
            # Ensure model is loaded first
            _ = self.model
        return self._layers

    @property
    def model(self):
        """Lazy load the Gemma model."""
        if self._model is None:
            print(f"Loading model from {self.model_path}...")
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
            self._model.eval()
            # Find and cache the layers path
            self._layers = self._find_layers(self._model)
            print(f"Model loaded successfully. Found {len(self._layers)} layers.")
        return self._model

    @property
    def tokenizer(self):
        """Lazy load the tokenizer."""
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        return self._tokenizer

    @property
    def saes(self) -> dict[int, JumpReLUSAE]:
        """Lazy load all SAEs for configured layers."""
        if not self._saes:
            print(f"Loading SAEs for layers {self.sae_layers} (width={self.sae_width})...")
            for layer in self.sae_layers:
                self._saes[layer] = self._load_sae(layer)
                print(f"  Layer {layer} SAE loaded.")
            print("All SAEs loaded successfully.")
        return self._saes

    def get_sae(self, layer: int) -> JumpReLUSAE:
        """Get SAE for a specific layer."""
        if layer not in self._saes:
            if layer not in self.sae_layers:
                raise ValueError(f"Layer {layer} not in available layers: {self.sae_layers}")
            self._saes[layer] = self._load_sae(layer)
        return self._saes[layer]

    def _load_sae(self, layer: int) -> JumpReLUSAE:
        """Download and load SAE weights from HuggingFace for a specific layer."""
        path_to_params = hf_hub_download(
            repo_id=SAE_REPO,
            filename=f"resid_post/layer_{layer}_width_{self.sae_width}_l0_{self.sae_l0}/params.safetensors",
        )
        params = load_file(path_to_params)

        d_model, d_sae = params["w_enc"].shape
        sae = JumpReLUSAE(d_model, d_sae)
        sae.load_state_dict(params)
        sae.to(self.device)
        sae.eval()

        return sae

    def _gather_acts_hook(self, mod, inputs, outputs, cache: dict, key: str):
        """Generic hook to capture layer outputs."""
        cache[key] = outputs[0].detach()
        return outputs

    def gather_residual_activations(self, input_ids: torch.Tensor, layers: list[int] = None) -> dict[int, torch.Tensor]:
        """
        Run model and capture residual stream activations at multiple layers.

        Args:
            input_ids: Tokenized input tensor of shape (1, seq_len)
            layers: List of layer indices to capture (defaults to all SAE layers)

        Returns:
            Dictionary mapping layer index to activations tensor of shape (seq_len, d_model)
        """
        if layers is None:
            layers = self.sae_layers

        cache = {}
        handles = []

        # Hook into all target layers
        for layer_idx in layers:
            handle = self.layers[layer_idx].register_forward_hook(
                partial(self._gather_acts_hook, cache=cache, key=f"layer_{layer_idx}")
            )
            handles.append(handle)

        try:
            _ = self.model.forward(input_ids)
        finally:
            for handle in handles:
                handle.remove()

        # Return dict: layer_idx -> (seq_len, d_model)
        return {
            layer_idx: cache[f"layer_{layer_idx}"].squeeze(0)
            for layer_idx in layers
        }

    def analyze_prompt(self, prompt: str, top_k: int = 10) -> dict:
        """
        Analyze a prompt and return activation data for all SAE layers.

        Args:
            prompt: The text prompt to analyze
            top_k: Number of top features to return per token

        Returns:
            Dictionary with tokens and per-layer activation data
        """
        # Tokenize
        inputs = self.tokenizer.encode(
            prompt,
            return_tensors="pt",
            add_special_tokens=True
        ).to(self.device)

        # Get string tokens for display
        str_tokens = self.tokenizer.convert_ids_to_tokens(inputs[0].tolist())
        # Clean up token display (replace special chars)
        str_tokens = [t.replace("▁", " ").replace("<0x0A>", "\\n") for t in str_tokens]

        # Gather residual activations from all SAE layers in one forward pass
        residuals_by_layer = self.gather_residual_activations(inputs)

        # Process each layer
        layers_data = {}
        for layer_idx in self.sae_layers:
            sae = self.get_sae(layer_idx)
            residuals = residuals_by_layer[layer_idx]

            # Encode with SAE
            sae_acts = sae.encode(residuals.to(torch.float32))
            # Shape: (seq_len, d_sae)

            # Get top features per token
            top_acts_per_token, top_features_per_token = sae_acts.topk(top_k, dim=-1)

            # Get globally top features (by mean activation across sequence)
            # Skip BOS token (index 0) as SAEs aren't trained on it
            mean_acts = sae_acts[1:].mean(dim=0)
            top_global_acts, top_global_features = mean_acts.topk(top_k)

            # Get max activating token for each global top feature
            global_feature_info = []
            for feat_idx, mean_act in zip(top_global_features.tolist(), top_global_acts.tolist()):
                feat_acts = sae_acts[:, feat_idx]
                max_pos = feat_acts.argmax().item()
                max_act = feat_acts[max_pos].item()
                global_feature_info.append({
                    "id": feat_idx,
                    "mean_activation": round(mean_act, 4),
                    "max_activation": round(max_act, 4),
                    "max_token_pos": max_pos,
                    "max_token": str_tokens[max_pos] if max_pos < len(str_tokens) else "?",
                })

            layers_data[layer_idx] = {
                "sae_acts": sae_acts.detach().cpu().numpy().tolist(),
                "top_features_per_token": top_features_per_token.detach().cpu().numpy().tolist(),
                "top_acts_per_token": top_acts_per_token.detach().cpu().numpy().tolist(),
                "top_features_global": global_feature_info,
                "num_features": sae_acts.shape[-1],
            }

        return {
            "tokens": str_tokens,
            "num_tokens": len(str_tokens),
            "layers": layers_data,
            "available_layers": self.sae_layers,
            "sae_config": {
                "width": self.sae_width,
                "l0": self.sae_l0,
            }
        }

    def analyze_prompt_lazy(self, prompt: str) -> dict:
        """
        Initial analysis that only tokenizes and returns available layers.
        Does NOT load any SAE weights - those are loaded on-demand via analyze_layer.

        Args:
            prompt: The text prompt to analyze

        Returns:
            Dictionary with tokens and available layers (no SAE data)
        """
        # Tokenize
        inputs = self.tokenizer.encode(
            prompt,
            return_tensors="pt",
            add_special_tokens=True
        ).to(self.device)

        # Get string tokens for display
        str_tokens = self.tokenizer.convert_ids_to_tokens(inputs[0].tolist())
        str_tokens = [t.replace("▁", " ").replace("<0x0A>", "\\n") for t in str_tokens]

        # Cache residual activations for later layer analysis
        cache_key = hash(prompt)
        if cache_key not in _residual_cache:
            residuals_by_layer = self.gather_residual_activations(inputs)
            _residual_cache[cache_key] = {
                "residuals": residuals_by_layer,
                "tokens": str_tokens,
                "inputs": inputs,
            }

        return {
            "tokens": str_tokens,
            "num_tokens": len(str_tokens),
            "available_layers": self.sae_layers,
            "sae_config": {
                "width": self.sae_width,
                "l0": self.sae_l0,
            }
        }

    def analyze_layer(self, prompt: str, layer: int, top_k: int = 10) -> dict:
        """
        Analyze a specific layer for a prompt. Loads SAE weights on-demand.
        Uses cached residual activations if available.

        Args:
            prompt: The text prompt to analyze
            layer: The SAE layer index to analyze
            top_k: Number of top features to return per token

        Returns:
            Dictionary with SAE activation data for this layer
        """
        if layer not in self.sae_layers:
            raise ValueError(f"Layer {layer} not in available layers: {self.sae_layers}")

        cache_key = hash(prompt)

        # Check if we have cached residuals
        if cache_key in _residual_cache:
            cached = _residual_cache[cache_key]
            residuals = cached["residuals"][layer]
            str_tokens = cached["tokens"]
        else:
            # Need to run model (shouldn't happen in normal flow)
            inputs = self.tokenizer.encode(
                prompt,
                return_tensors="pt",
                add_special_tokens=True
            ).to(self.device)

            str_tokens = self.tokenizer.convert_ids_to_tokens(inputs[0].tolist())
            str_tokens = [t.replace("▁", " ").replace("<0x0A>", "\\n") for t in str_tokens]

            residuals_by_layer = self.gather_residual_activations(inputs, layers=[layer])
            residuals = residuals_by_layer[layer]

            # Cache for future layer requests
            if cache_key not in _residual_cache:
                # Run full model to cache all layers
                full_residuals = self.gather_residual_activations(inputs)
                _residual_cache[cache_key] = {
                    "residuals": full_residuals,
                    "tokens": str_tokens,
                    "inputs": inputs,
                }

        # Load SAE for this layer (lazy - only loads if not already loaded)
        sae = self.get_sae(layer)

        # Encode with SAE
        sae_acts = sae.encode(residuals.to(torch.float32))

        # Get top features per token
        top_acts_per_token, top_features_per_token = sae_acts.topk(top_k, dim=-1)

        # Get globally top features (by mean activation across sequence)
        mean_acts = sae_acts[1:].mean(dim=0)  # Skip BOS
        top_global_acts, top_global_features = mean_acts.topk(top_k)

        # Get max activating token for each global top feature
        global_feature_info = []
        for feat_idx, mean_act in zip(top_global_features.tolist(), top_global_acts.tolist()):
            feat_acts = sae_acts[:, feat_idx]
            max_pos = feat_acts.argmax().item()
            max_act = feat_acts[max_pos].item()
            global_feature_info.append({
                "id": feat_idx,
                "mean_activation": round(mean_act, 4),
                "max_activation": round(max_act, 4),
                "max_token_pos": max_pos,
                "max_token": str_tokens[max_pos] if max_pos < len(str_tokens) else "?",
            })

        return {
            "layer": layer,
            "sae_acts": sae_acts.detach().cpu().numpy().tolist(),
            "top_features_per_token": top_features_per_token.detach().cpu().numpy().tolist(),
            "top_acts_per_token": top_acts_per_token.detach().cpu().numpy().tolist(),
            "top_features_global": global_feature_info,
            "num_features": sae_acts.shape[-1],
        }

    def clear_residual_cache(self):
        """Clear the residual activation cache."""
        global _residual_cache
        _residual_cache.clear()

    def get_feature_activations(self, prompt: str, feature_id: int, layer: int = None) -> dict:
        """
        Get activation pattern for a specific feature across all tokens.

        Args:
            prompt: The text prompt
            feature_id: The SAE feature index
            layer: The SAE layer (defaults to first available layer)

        Returns:
            Dictionary with per-token activations for the feature
        """
        if layer is None:
            layer = self.sae_layers[0]

        # Tokenize
        inputs = self.tokenizer.encode(
            prompt,
            return_tensors="pt",
            add_special_tokens=True
        ).to(self.device)

        str_tokens = self.tokenizer.convert_ids_to_tokens(inputs[0].tolist())
        str_tokens = [t.replace("▁", " ").replace("<0x0A>", "\\n") for t in str_tokens]

        # Get activations for specific layer
        residuals_by_layer = self.gather_residual_activations(inputs, layers=[layer])
        sae = self.get_sae(layer)
        sae_acts = sae.encode(residuals_by_layer[layer].to(torch.float32))

        # Get activations for specific feature
        feature_acts = sae_acts[:, feature_id].detach().cpu().numpy()
        max_act = float(feature_acts.max()) + 1e-6

        return {
            "feature_id": feature_id,
            "layer": layer,
            "tokens": str_tokens,
            "activations": feature_acts.tolist(),
            "normalized_activations": (feature_acts / max_act).tolist(),
            "max_activation": float(feature_acts.max()),
            "mean_activation": float(feature_acts[1:].mean()),  # Skip BOS
        }

    def generate_with_steering(
        self,
        prompt: str,
        steering_features: list[dict],
        max_new_tokens: int = 80,
        steering_layer_offset: int = -5,
        sae_layer: int = None,
    ) -> dict:
        """
        Generate text with feature steering applied.

        Args:
            prompt: The input prompt
            steering_features: List of {"feature_id": int, "coefficient": float, "layer": int (optional)}
            max_new_tokens: Maximum tokens to generate
            steering_layer_offset: Layer offset from SAE layer for steering
            sae_layer: Default SAE layer for features without explicit layer

        Returns:
            Dictionary with original and steered outputs
        """
        if sae_layer is None:
            sae_layer = self.sae_layers[0]

        # Format prompt for chat
        messages = [{"role": "user", "content": prompt}]
        formatted = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.tokenizer.encode(
            formatted,
            return_tensors="pt",
            add_special_tokens=True
        ).to(self.device)

        # Generate without steering first
        with torch.no_grad():
            outputs_clean = self.model.generate(
                input_ids=inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        original_output = self.tokenizer.decode(
            outputs_clean[0][inputs.shape[1]:],
            skip_special_tokens=True
        )

        if not steering_features:
            return {
                "prompt": prompt,
                "original_output": original_output,
                "steered_output": original_output,
                "steering_applied": [],
            }

        # Group steering features by layer
        features_by_layer = {}
        for sf in steering_features:
            layer = sf.get("layer", sae_layer)
            if layer not in features_by_layer:
                features_by_layer[layer] = []
            features_by_layer[layer].append(sf)

        # Create steering hooks for each layer
        handles = []

        for layer, layer_features in features_by_layer.items():
            sae = self.get_sae(layer)
            steering_target_layer = layer + steering_layer_offset

            def make_steering_hook(sae_ref, feats):
                def steering_hook(mod, inputs, outputs):
                    output = outputs[0]
                    for sf in feats:
                        feat_idx = sf["feature_id"]
                        coeff = sf["coefficient"]
                        decoder_vec = sae_ref.w_dec[feat_idx]

                        # Handle KV caching - different logic for first vs cached passes
                        if output.shape[1] == 1:
                            avg_norm = torch.norm(output, dim=-1)
                            output += coeff * avg_norm * decoder_vec
                        else:
                            avg_norm = torch.norm(output[0, 1:], dim=-1, keepdim=True)
                            output[0, 1:] += coeff * avg_norm * decoder_vec

                    return outputs
                return steering_hook

            handle = self.layers[steering_target_layer].register_forward_hook(
                make_steering_hook(sae, layer_features)
            )
            handles.append(handle)

        # Generate with steering
        try:
            with torch.no_grad():
                outputs_steered = self.model.generate(
                    input_ids=inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
            steered_output = self.tokenizer.decode(
                outputs_steered[0][inputs.shape[1]:],
                skip_special_tokens=True
            )
        finally:
            for handle in handles:
                handle.remove()

        return {
            "prompt": prompt,
            "original_output": original_output,
            "steered_output": steered_output,
            "steering_applied": steering_features,
            "steering_layers": list(features_by_layer.keys()),
        }

    def _find_lm_head(self):
        """Find the language model head for unembedding."""
        # Try different paths for different architectures
        if hasattr(self.model, 'lm_head'):
            return self.model.lm_head
        if hasattr(self.model, 'language_model') and hasattr(self.model.language_model, 'lm_head'):
            return self.model.language_model.lm_head
        raise AttributeError("Cannot find lm_head in model")

    def get_top_logits_for_feature(self, feature_id: int, layer: int = None, top_k: int = 10) -> list[dict]:
        """
        Get the top predicted tokens when this feature fires (via unembedding).

        Args:
            feature_id: The SAE feature index
            layer: The SAE layer (defaults to first available layer)
            top_k: Number of top tokens to return

        Returns:
            List of {"token": str, "logit": float}
        """
        if layer is None:
            layer = self.sae_layers[0]

        # Get the decoder vector for this feature
        sae = self.get_sae(layer)
        decoder_vec = sae.w_dec[feature_id].detach().float()

        # Find and get the unembedding matrix
        lm_head = self._find_lm_head()
        w_u = lm_head.weight.detach().float()  # (vocab_size, d_model)

        # Project decoder vector through unembedding
        # decoder_vec shape: (d_model,) -> logits shape: (vocab_size,)
        logits = w_u @ decoder_vec

        # Get more candidates to filter through
        top_logits, top_indices = logits.topk(top_k * 10)

        results = []
        for logit, idx in zip(top_logits.tolist(), top_indices.tolist()):
            token = self.tokenizer.decode([idx])

            # Filter out garbage tokens
            if self._is_garbage_token(token):
                continue

            results.append({
                "token": token,
                "token_id": idx,
                "logit": round(logit, 4),
            })

            if len(results) >= top_k:
                break

        return results

    def _is_garbage_token(self, token: str) -> bool:
        """Check if a token is likely garbage/unused."""
        # Empty tokens
        if not token:
            return True

        # Special tokens
        if token.startswith('<') and token.endswith('>'):
            return True

        # Replacement character
        if '\ufffd' in token:
            return True

        # Only allow tokens that are mostly ASCII/Latin
        ascii_count = 0
        total_count = 0
        for char in token:
            total_count += 1
            code = ord(char)
            # Count as "good" if ASCII printable or common Latin
            if 32 <= code < 127:  # ASCII printable
                ascii_count += 1
            elif 0x00C0 <= code <= 0x00FF:  # Latin-1 Supplement (accents)
                ascii_count += 1
            elif 0x0100 <= code <= 0x017F:  # Latin Extended-A
                ascii_count += 1

        # Require at least 70% ASCII/Latin characters
        if total_count > 0 and ascii_count / total_count < 0.7:
            return True

        return False

    # =========================================================================
    # Refusal Pathway Analysis Methods
    # =========================================================================

    def get_neuronpedia_model_id(self) -> str:
        """Get the Neuronpedia model ID based on SAE repo."""
        # Map SAE repo to Neuronpedia model ID
        if "12b" in SAE_REPO.lower():
            return "gemma-3-12b-it"
        elif "4b" in SAE_REPO.lower():
            return "gemma-3-4b-it"
        return "gemma-3-12b-it"  # Default

    def get_neuronpedia_source_id(self, layer: int) -> str:
        """Get the Neuronpedia source/SAE ID for a layer."""
        # Format: layer-{layer}-gemmascope-res-{width}
        return f"{layer}-gemmascope-2-res-262k"

    def get_neuronpedia_url(self, feature_id: int, layer: int) -> str:
        """Generate Neuronpedia URL for a feature."""
        model_id = self.get_neuronpedia_model_id()
        source_id = self.get_neuronpedia_source_id(layer)
        return f"https://www.neuronpedia.org/{model_id}/{source_id}/{feature_id}"

    def get_neuronpedia_embed_url(self, feature_id: int, layer: int) -> str:
        """Generate embeddable Neuronpedia URL with embed options."""
        base_url = self.get_neuronpedia_url(feature_id, layer)
        # Using only documented embed parameters
        embed_params = [
            "embed=true",
            "embedexplanation=true",
            "embedplots=true",
            "embedtest=true",
        ]
        return f"{base_url}?{'&'.join(embed_params)}"

    def has_neuronpedia_data(self, layer: int) -> bool:
        """Check if Neuronpedia has data for this layer."""
        return layer in NEURONPEDIA_LAYERS

    def fetch_neuronpedia_data(self, feature_id: int, layer: int) -> dict:
        """
        Fetch feature data from Neuronpedia API.

        Returns explanations, lists, positive/negative logits, and top activations.
        """
        # Check if Neuronpedia has data for this layer
        if not self.has_neuronpedia_data(layer):
            return {
                "error": f"Neuronpedia does not have data for layer {layer}",
                "unsupported_layer": True,
                "supported_layers": NEURONPEDIA_LAYERS,
            }

        model_id = self.get_neuronpedia_model_id()
        source_id = self.get_neuronpedia_source_id(layer)
        cache_key = f"{model_id}/{source_id}/{feature_id}"

        # Check cache first
        if cache_key in _neuronpedia_cache:
            return _neuronpedia_cache[cache_key]

        if not NEURONPEDIA_API_KEY:
            return {"error": "No Neuronpedia API key configured"}

        try:
            url = f"{NEURONPEDIA_BASE_URL}/feature/{model_id}/{source_id}/{feature_id}"
            headers = {"x-api-key": NEURONPEDIA_API_KEY}
            response = requests.get(url, headers=headers, timeout=10)

            if response.status_code == 404:
                return {"error": "Feature not found on Neuronpedia"}

            response.raise_for_status()
            data = response.json()

            # Extract relevant fields
            result = {
                "feature_id": feature_id,
                "layer": layer,
                "neuronpedia_url": self.get_neuronpedia_url(feature_id, layer),
                "neuronpedia_embed_url": self.get_neuronpedia_embed_url(feature_id, layer),
                # Explanations
                "explanations": [],
                # Lists/categories
                "lists": [],
                # Positive and negative logits (token associations)
                "positive_logits": [],
                "negative_logits": [],
                # Top activations (example texts)
                "top_activations": [],
                # Statistics
                "max_activation": data.get("maxActApprox"),
                "frac_nonzero": data.get("frac_nonzero"),
                # Activation histogram
                "activation_histogram": {
                    "heights": data.get("freq_hist_data_bar_heights", []),
                    "values": data.get("freq_hist_data_bar_values", []),
                },
            }

            # Extract explanations
            if "explanations" in data and data["explanations"]:
                for exp in data["explanations"]:
                    result["explanations"].append({
                        "description": exp.get("description", ""),
                        "score": exp.get("score"),
                        "model": exp.get("explanationModelName"),
                    })

            # Extract positive/negative logits
            if "pos_str" in data and "pos_values" in data:
                for token, value in zip(data["pos_str"], data["pos_values"]):
                    result["positive_logits"].append({"token": token, "value": value})

            if "neg_str" in data and "neg_values" in data:
                for token, value in zip(data["neg_str"], data["neg_values"]):
                    result["negative_logits"].append({"token": token, "value": value})

            # Extract top activations (example texts where feature fires)
            if "activations" in data and data["activations"]:
                for act in data["activations"][:10]:  # Limit to 10 examples
                    tokens = act.get("tokens", [])
                    values = act.get("values", [])
                    max_value = act.get("maxValue", 0)
                    max_idx = act.get("maxValueTokenIndex", 0)

                    result["top_activations"].append({
                        "tokens": tokens,
                        "values": values,
                        "max_value": max_value,
                        "max_token_index": max_idx,
                    })

            # Cache the result
            _neuronpedia_cache[cache_key] = result
            return result

        except requests.exceptions.Timeout:
            return {"error": "Neuronpedia API timeout"}
        except requests.exceptions.RequestException as e:
            return {"error": f"Neuronpedia API error: {str(e)}"}
        except Exception as e:
            return {"error": f"Failed to parse Neuronpedia data: {str(e)}"}

    def compare_prompts(
        self,
        prompt_a: str,
        prompt_b: str,
        top_k: int = 50,
        threshold: float = 0.1
    ) -> dict:
        """
        Compare activations between two prompts and return differential features.

        Args:
            prompt_a: First prompt (typically harmful/refused)
            prompt_b: Second prompt (typically benign/accepted)
            top_k: Number of top differential features per layer
            threshold: Minimum activation to consider

        Returns:
            Dictionary with differential features per layer
        """
        # Tokenize both prompts
        inputs_a = self.tokenizer.encode(
            prompt_a, return_tensors="pt", add_special_tokens=True
        ).to(self.device)
        inputs_b = self.tokenizer.encode(
            prompt_b, return_tensors="pt", add_special_tokens=True
        ).to(self.device)

        tokens_a = self.tokenizer.convert_ids_to_tokens(inputs_a[0].tolist())
        tokens_b = self.tokenizer.convert_ids_to_tokens(inputs_b[0].tolist())
        tokens_a = [t.replace("▁", " ").replace("<0x0A>", "\\n") for t in tokens_a]
        tokens_b = [t.replace("▁", " ").replace("<0x0A>", "\\n") for t in tokens_b]

        # Gather activations for both prompts
        residuals_a = self.gather_residual_activations(inputs_a)
        residuals_b = self.gather_residual_activations(inputs_b)

        layers_data = {}

        for layer_idx in self.sae_layers:
            sae = self.get_sae(layer_idx)

            # Encode with SAE
            sae_acts_a = sae.encode(residuals_a[layer_idx].to(torch.float32))
            sae_acts_b = sae.encode(residuals_b[layer_idx].to(torch.float32))

            # Compute mean activations (skip BOS token)
            mean_acts_a = sae_acts_a[1:].mean(dim=0)
            mean_acts_b = sae_acts_b[1:].mean(dim=0)

            # Compute differential
            diff = mean_acts_a - mean_acts_b
            epsilon = 1e-6
            ratio = mean_acts_a / (mean_acts_b + epsilon)

            # Get features above threshold in either prompt
            mask = (mean_acts_a > threshold) | (mean_acts_b > threshold)
            valid_indices = torch.where(mask)[0]

            if len(valid_indices) == 0:
                layers_data[layer_idx] = {"differential_features": []}
                continue

            # Get differential values for valid features
            valid_diffs = diff[valid_indices]
            valid_acts_a = mean_acts_a[valid_indices]
            valid_acts_b = mean_acts_b[valid_indices]
            valid_ratios = ratio[valid_indices]

            # Sort by absolute differential (most different first)
            sorted_indices = torch.argsort(valid_diffs.abs(), descending=True)[:top_k]

            differential_features = []
            for idx in sorted_indices:
                feat_idx = valid_indices[idx].item()
                differential_features.append({
                    "feature_id": feat_idx,
                    "mean_diff": round(valid_diffs[idx].item(), 4),
                    "activation_a": round(valid_acts_a[idx].item(), 4),
                    "activation_b": round(valid_acts_b[idx].item(), 4),
                    "ratio": round(valid_ratios[idx].item(), 4),
                    "neuronpedia_url": self.get_neuronpedia_url(feat_idx, layer_idx),
                })

            # Get top feature IDs for indexing activations
            top_feature_ids = [f["feature_id"] for f in differential_features]

            # Store per-token activations for top features (sparse format)
            # Format: {feature_id: [act_per_token, ...]}
            token_acts_a = {}
            token_acts_b = {}
            for feat_id in top_feature_ids:
                token_acts_a[feat_id] = [round(v, 4) for v in sae_acts_a[:, feat_id].tolist()]
                token_acts_b[feat_id] = [round(v, 4) for v in sae_acts_b[:, feat_id].tolist()]

            layers_data[layer_idx] = {
                "differential_features": differential_features,
                "token_activations_a": token_acts_a,
                "token_activations_b": token_acts_b,
            }

        return {
            "prompt_a": prompt_a,
            "prompt_b": prompt_b,
            "tokens_a": tokens_a,
            "tokens_b": tokens_b,
            "layers": layers_data,
            "available_layers": self.sae_layers,
        }

    # Default refusal phrases to detect
    REFUSAL_PHRASES = [
        "I can't", "I cannot", "I'm sorry", "I apologize",
        "I'm not able", "I am not able", "I won't", "I will not",
        "I'm unable", "I am unable", "As an AI",
        "I must decline", "I cannot assist", "I can't help",
        "against my guidelines", "harmful", "unethical",
        "I'm not going to", "I refuse",
    ]

    def generate_and_detect_refusal(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        refusal_phrases: list[str] = None,
    ) -> dict:
        """
        Generate a response and detect refusal patterns.

        Args:
            prompt: Input prompt
            max_new_tokens: Max tokens to generate
            refusal_phrases: Custom refusal phrases (uses defaults if None)

        Returns:
            Dictionary with refusal detection and correlated features
        """
        if refusal_phrases is None:
            refusal_phrases = self.REFUSAL_PHRASES

        # Format prompt for chat
        messages = [{"role": "user", "content": prompt}]
        formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer.encode(
            formatted, return_tensors="pt", add_special_tokens=True
        ).to(self.device)
        prompt_len = inputs.shape[1]

        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        # Decode generated text
        generated_ids = outputs[0][prompt_len:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Detect refusal phrases
        refusal_detected = False
        phrases_found = []
        for phrase in refusal_phrases:
            if phrase.lower() in generated_text.lower():
                refusal_detected = True
                phrases_found.append(phrase)

        # Get token-level info for generated text
        full_tokens = self.tokenizer.convert_ids_to_tokens(outputs[0].tolist())
        full_tokens = [t.replace("▁", " ").replace("<0x0A>", "\\n") for t in full_tokens]
        generated_tokens = full_tokens[prompt_len:]

        # Find token positions of refusal phrases (approximate)
        refusal_positions = []
        gen_text_lower = generated_text.lower()
        for phrase in phrases_found:
            pos = gen_text_lower.find(phrase.lower())
            if pos >= 0:
                # Approximate token position (not exact but useful)
                char_count = 0
                for i, tok in enumerate(generated_tokens):
                    char_count += len(tok)
                    if char_count >= pos:
                        refusal_positions.append(prompt_len + i)
                        break

        # Analyze activations for the full sequence
        residuals = self.gather_residual_activations(outputs)

        layers_data = {}
        for layer_idx in self.sae_layers:
            sae = self.get_sae(layer_idx)
            sae_acts = sae.encode(residuals[layer_idx].to(torch.float32))

            # Compare refusal positions to overall mean
            gen_acts = sae_acts[prompt_len:]  # Only generated tokens
            mean_gen_acts = gen_acts.mean(dim=0)

            # If we have refusal positions, compare them
            if refusal_positions:
                refusal_indices = [p - prompt_len for p in refusal_positions if p - prompt_len < len(gen_acts)]
                if refusal_indices:
                    refusal_acts = gen_acts[refusal_indices].mean(dim=0)
                    # Score = how much higher at refusal vs overall
                    correlation_scores = refusal_acts - mean_gen_acts
                else:
                    correlation_scores = torch.zeros_like(mean_gen_acts)
            else:
                correlation_scores = torch.zeros_like(mean_gen_acts)

            # Get top correlated features
            top_scores, top_indices = correlation_scores.topk(min(50, len(correlation_scores)))

            refusal_features = []
            for score, feat_idx in zip(top_scores.tolist(), top_indices.tolist()):
                if score > 0.01:  # Only include meaningfully correlated
                    refusal_features.append({
                        "feature_id": feat_idx,
                        "correlation_score": round(score, 4),
                        "mean_activation": round(mean_gen_acts[feat_idx].item(), 4),
                        "neuronpedia_url": self.get_neuronpedia_url(feat_idx, layer_idx),
                    })

            layers_data[layer_idx] = {
                "refusal_correlated_features": refusal_features[:20],
            }

        return {
            "prompt": prompt,
            "generated_text": generated_text,
            "refusal_detected": refusal_detected,
            "refusal_phrases_found": phrases_found,
            "generated_tokens": generated_tokens,
            "refusal_token_positions": refusal_positions,
            "layers": layers_data,
            "available_layers": self.sae_layers,
        }

    def rank_features_for_refusal(
        self,
        prompt_pairs: list[dict],
        top_k: int = 100,
    ) -> dict:
        """
        Rank features by their correlation with harmful vs benign prompts.

        Args:
            prompt_pairs: List of {"harmful": str, "benign": str} dicts
            top_k: Number of top features to return per layer

        Returns:
            Dictionary with ranked features per layer
        """
        # Initialize accumulators per layer
        harmful_sums = {layer: None for layer in self.sae_layers}
        benign_sums = {layer: None for layer in self.sae_layers}
        diff_positive_counts = {layer: None for layer in self.sae_layers}
        num_pairs = len(prompt_pairs)

        for pair in prompt_pairs:
            harmful = pair.get("harmful", "")
            benign = pair.get("benign", "")

            # Tokenize
            inputs_h = self.tokenizer.encode(
                harmful, return_tensors="pt", add_special_tokens=True
            ).to(self.device)
            inputs_b = self.tokenizer.encode(
                benign, return_tensors="pt", add_special_tokens=True
            ).to(self.device)

            # Get activations
            residuals_h = self.gather_residual_activations(inputs_h)
            residuals_b = self.gather_residual_activations(inputs_b)

            for layer_idx in self.sae_layers:
                sae = self.get_sae(layer_idx)

                acts_h = sae.encode(residuals_h[layer_idx].to(torch.float32))
                acts_b = sae.encode(residuals_b[layer_idx].to(torch.float32))

                mean_h = acts_h[1:].mean(dim=0)  # Skip BOS
                mean_b = acts_b[1:].mean(dim=0)

                # Initialize accumulators on first pair
                if harmful_sums[layer_idx] is None:
                    harmful_sums[layer_idx] = torch.zeros_like(mean_h)
                    benign_sums[layer_idx] = torch.zeros_like(mean_b)
                    diff_positive_counts[layer_idx] = torch.zeros_like(mean_h)

                harmful_sums[layer_idx] += mean_h
                benign_sums[layer_idx] += mean_b
                diff_positive_counts[layer_idx] += (mean_h > mean_b).float()

        # Compute rankings
        layers_data = {}
        for layer_idx in self.sae_layers:
            if harmful_sums[layer_idx] is None:
                layers_data[layer_idx] = {"ranked_features": []}
                continue

            mean_harmful = harmful_sums[layer_idx] / num_pairs
            mean_benign = benign_sums[layer_idx] / num_pairs
            consistency = diff_positive_counts[layer_idx] / num_pairs

            # Differential score: consistency * |mean_diff|
            mean_diff = mean_harmful - mean_benign
            differential_score = consistency * mean_diff.abs()

            # Get top features by differential score
            top_scores, top_indices = differential_score.topk(top_k)

            ranked_features = []
            for score, feat_idx in zip(top_scores.tolist(), top_indices.tolist()):
                ranked_features.append({
                    "feature_id": feat_idx,
                    "consistency_score": round(consistency[feat_idx].item(), 4),
                    "mean_harmful_activation": round(mean_harmful[feat_idx].item(), 4),
                    "mean_benign_activation": round(mean_benign[feat_idx].item(), 4),
                    "differential_score": round(score, 4),
                    "neuronpedia_url": self.get_neuronpedia_url(feat_idx, layer_idx),
                })

            layers_data[layer_idx] = {"ranked_features": ranked_features}

        return {
            "num_prompt_pairs": num_pairs,
            "layers": layers_data,
            "available_layers": self.sae_layers,
        }


# =============================================================================
# Singleton instance for Flask app
# =============================================================================

_manager: Optional[SAEModelManager] = None


def get_manager() -> SAEModelManager:
    """Get or create the global model manager instance."""
    global _manager
    if _manager is None:
        _manager = SAEModelManager()
    return _manager


def initialize_models():
    """Pre-load all models (call at startup to avoid first-request delay)."""
    manager = get_manager()
    # Access properties to trigger lazy loading
    _ = manager.model
    _ = manager.saes  # Load all SAEs
    print("All models initialized.")
