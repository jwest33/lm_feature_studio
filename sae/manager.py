"""
SAE Model Manager

Core manager class that handles model loading and SAE management.
Composed with mixins for analysis, steering, comparison, refusal, and Neuronpedia features.
"""

import gc
import torch
from functools import partial
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from .config import (
    MODEL_PATH,
    SAE_REPO,
    SAE_WIDTH,
    SAE_L0,
    BASE_MODEL,
    get_available_layers,
    get_neuronpedia_layers,
)
from .sae import JumpReLUSAE
from .cache import (
    _residual_cache,
    _neuronpedia_cache,
    clear_residual_cache,
    get_memory_status,
)
from .analysis import AnalysisMixin
from .steering import SteeringMixin
from .comparison import ComparisonMixin
from .refusal import RefusalMixin
from .neuronpedia import NeuronpediaMixin


class SAEModelManager(
    AnalysisMixin,
    SteeringMixin,
    ComparisonMixin,
    RefusalMixin,
    NeuronpediaMixin,
):
    """
    Manages the Gemma model and SAE for the Flask app.

    Handles lazy loading to avoid loading models on import.
    """

    def __init__(
        self,
        model_path: str = MODEL_PATH,
        sae_repo: str = SAE_REPO,
        sae_layers: list[int] = None,
        sae_width: str = SAE_WIDTH,
        sae_l0: str = SAE_L0,
        base_model: str = BASE_MODEL,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model_path = model_path
        self.sae_repo = sae_repo
        self.base_model = base_model
        self.sae_layers = sae_layers if sae_layers is not None else get_available_layers(base_model)
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

    def unload(self):
        """Unload all models and clear caches to free memory."""
        # Clear SAEs
        for sae in self._saes.values():
            del sae
        self._saes.clear()

        # Clear model
        if self._model is not None:
            del self._model
            self._model = None

        # Clear tokenizer
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None

        self._layers = None

        # Clear caches
        _residual_cache.clear()
        _neuronpedia_cache.clear()

        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("Models unloaded and caches cleared.")

    def unload_llm(self):
        """
        Unload only the LLM to free GPU memory, preserving SAEs and caches.

        Use this after gathering residuals to free memory before loading SAEs.
        The tokenizer is kept loaded as it uses minimal memory.
        """
        if self._model is not None:
            del self._model
            self._model = None
            self._layers = None

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print("LLM unloaded. Residual cache preserved.")
        else:
            print("LLM was not loaded.")

    def unload_all_saes(self):
        """Unload all SAEs to free GPU memory, preserving LLM and caches."""
        for sae in self._saes.values():
            del sae
        self._saes.clear()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("All SAEs unloaded.")

    def reconfigure(
        self,
        model_path: str = None,
        sae_repo: str = None,
        sae_width: str = None,
        sae_l0: str = None,
        base_model: str = None,
    ) -> dict:
        """
        Reconfigure the manager with new settings.

        Args:
            model_path: New model path (local or HuggingFace ID)
            sae_repo: New SAE repository
            sae_width: New SAE width (16k, 65k, 262k, 1M)
            sae_l0: New SAE L0 (small, medium, large)
            base_model: Base model size ("4b" or "12b") - controls SAE layers

        Returns:
            Dictionary with new configuration
        """
        # Unload current models
        self.unload()

        # Update configuration
        if model_path is not None:
            self.model_path = model_path
        if sae_repo is not None:
            self.sae_repo = sae_repo
        if sae_width is not None:
            self.sae_width = sae_width
        if sae_l0 is not None:
            self.sae_l0 = sae_l0
        if base_model is not None:
            self.base_model = base_model
            # Update layers based on new base model
            self.sae_layers = get_available_layers(base_model)

        return {
            "model_path": self.model_path,
            "sae_repo": self.sae_repo,
            "sae_layers": self.sae_layers,
            "sae_width": self.sae_width,
            "sae_l0": self.sae_l0,
            "base_model": self.base_model,
            "device": self.device,
            "neuronpedia_layers": get_neuronpedia_layers(self.base_model),
        }

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

    def _ensure_sae_downloaded(self, layer: int) -> str:
        """
        Ensure SAE weights are downloaded for a layer (without loading into memory).

        Returns:
            Path to the downloaded safetensors file
        """
        print(f"  Downloading SAE for layer {layer} to disk (not loading into GPU)...")
        path = hf_hub_download(
            repo_id=self.sae_repo,
            filename=f"resid_post/layer_{layer}_width_{self.sae_width}_l0_{self.sae_l0}/params.safetensors",
        )
        return path

    def _ensure_all_saes_downloaded(self):
        """
        Ensure all SAE weights are downloaded before loading the LLM.

        This prevents the scenario where the LLM is loaded into memory
        but SAE download fails, wasting GPU memory.
        """
        print(f"Pre-downloading SAE files for layers {self.sae_layers} (disk only, not GPU)...")
        for layer in self.sae_layers:
            self._ensure_sae_downloaded(layer)
        print("All SAE files downloaded to disk. SAEs will be loaded into GPU on-demand.")

    @property
    def model(self):
        """Lazy load the Gemma model (after ensuring SAEs are downloaded)."""
        if self._model is None:
            # Download SAE weights BEFORE loading LLM to avoid wasting memory
            # if SAE download fails
            self._ensure_all_saes_downloaded()

            print(f"Loading model from {self.model_path}...")
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation="sdpa",
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

    def unload_sae(self, layer: int):
        """Unload SAE for a specific layer to free memory."""
        if layer in self._saes:
            del self._saes[layer]
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"Unloaded SAE for layer {layer}")

    def _load_sae(self, layer: int) -> JumpReLUSAE:
        """Download and load SAE weights from HuggingFace for a specific layer."""
        print(f"Loading SAE for layer {layer} into GPU memory...")
        path_to_params = hf_hub_download(
            repo_id=self.sae_repo,
            filename=f"resid_post/layer_{layer}_width_{self.sae_width}_l0_{self.sae_l0}/params.safetensors",
        )
        params = load_file(path_to_params)

        d_model, d_sae = params["w_enc"].shape
        sae = JumpReLUSAE(d_model, d_sae)
        sae.load_state_dict(params)
        sae.to(self.device)
        sae.eval()

        print(f"  Layer {layer} SAE loaded ({d_sae:,} features, {d_model} d_model)")
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

    def get_memory_status(self) -> dict:
        """
        Get current memory status for debugging sequential loading.

        Returns:
            Dictionary with memory usage information
        """
        return get_memory_status(self.device, self._saes, self._model)

    def clear_residual_cache(self):
        """Clear the residual activation cache."""
        clear_residual_cache()


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
