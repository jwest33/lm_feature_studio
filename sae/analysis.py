"""
Analysis Mixin

Methods for analyzing prompts and extracting SAE activations.
"""

import torch
from .cache import _residual_cache


class AnalysisMixin:
    """Mixin providing prompt analysis methods for SAEModelManager."""

    def gather_and_cache_residuals(
        self,
        prompt: str,
        unload_llm_after: bool = True,
        cache_key: str = None,
    ) -> dict:
        """
        Run LLM forward pass, cache residuals to CPU, and optionally unload LLM.

        This is the first step in the sequential workflow where LLM and SAE
        are never loaded simultaneously to save GPU memory.

        Args:
            prompt: The text prompt to analyze
            unload_llm_after: If True, unload LLM after caching residuals (default True)
            cache_key: Optional custom cache key (defaults to hash of prompt)

        Returns:
            Dictionary with cache_key, tokens, and available layers
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

        # Gather residual activations (requires LLM to be loaded)
        residuals_by_layer = self.gather_residual_activations(inputs)

        # Move residuals to CPU to free GPU memory
        residuals_cpu = {
            layer: tensor.cpu() for layer, tensor in residuals_by_layer.items()
        }

        # Cache with CPU tensors
        # Use integer hash keys for consistency with rank_features_layer
        if cache_key is None:
            cache_key = hash(prompt)
        elif isinstance(cache_key, str):
            cache_key = int(cache_key)

        _residual_cache[cache_key] = {
            "residuals": residuals_cpu,
            "tokens": str_tokens,
            "prompt": prompt,
            "on_cpu": True,  # Flag indicating tensors are on CPU
        }

        # Optionally unload LLM to free GPU memory for SAE
        if unload_llm_after:
            self.unload_llm()

        return {
            "cache_key": str(cache_key),  # String to avoid JS integer precision loss
            "tokens": str_tokens,
            "num_tokens": len(str_tokens),
            "available_layers": self.sae_layers,
            "llm_unloaded": unload_llm_after,
        }

    def gather_and_cache_residuals_batch(
        self,
        prompts: list[str],
        unload_llm_after: bool = True,
    ) -> dict:
        """
        Run LLM forward pass for multiple prompts, cache all residuals to CPU.

        More efficient than calling gather_and_cache_residuals() multiple times
        as it keeps the LLM loaded until all prompts are processed.

        Args:
            prompts: List of text prompts to analyze
            unload_llm_after: If True, unload LLM after caching all residuals

        Returns:
            Dictionary with cache_keys for each prompt
        """
        cache_keys = []

        for prompt in prompts:
            # Don't unload LLM until we're done with all prompts
            result = self.gather_and_cache_residuals(
                prompt,
                unload_llm_after=False,
                cache_key=None,
            )
            cache_keys.append(result["cache_key"])

        # Unload LLM after processing all prompts
        if unload_llm_after:
            self.unload_llm()

        return {
            "cache_keys": cache_keys,
            "num_prompts": len(prompts),
            "available_layers": self.sae_layers,
            "llm_unloaded": unload_llm_after,
        }

    def encode_cached_residuals(
        self,
        cache_key: int | str,
        layer: int,
        top_k: int = 10,
        unload_sae_after: bool = False,
    ) -> dict:
        """
        Load SAE and encode cached residuals for a specific layer.

        This is the second step in the sequential workflow. Call this after
        gather_and_cache_residuals() to get SAE activations without having
        the LLM in memory.

        Args:
            cache_key: Cache key from gather_and_cache_residuals() (string or int)
            layer: The SAE layer to analyze
            top_k: Number of top features to return per token
            unload_sae_after: If True, unload SAE after encoding (default False)

        Returns:
            Dictionary with SAE activation data for this layer
        """
        # Convert string cache key back to int (handles JS integer precision loss)
        if isinstance(cache_key, str):
            cache_key = int(cache_key)

        if cache_key not in _residual_cache:
            raise ValueError(f"Cache key '{cache_key}' not found. Run gather_and_cache_residuals first.")

        if layer not in self.sae_layers:
            raise ValueError(f"Layer {layer} not in available layers: {self.sae_layers}")

        cached = _residual_cache[cache_key]
        residuals_cpu = cached["residuals"][layer]
        str_tokens = cached["tokens"]

        # Load SAE for this layer (unload others to save memory)
        layers_to_unload = [l for l in list(self._saes.keys()) if l != layer]
        for l in layers_to_unload:
            self.unload_sae(l)

        sae = self.get_sae(layer)

        # Move residuals to GPU for SAE encoding
        residuals_gpu = residuals_cpu.to(self.device)

        # Encode with SAE
        sae_acts = sae.encode(residuals_gpu.to(torch.float32))

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

        # Clean up GPU tensor
        del residuals_gpu

        # Optionally unload SAE
        if unload_sae_after:
            self.unload_sae(layer)

        return {
            "layer": layer,
            "tokens": str_tokens,
            "sae_acts": sae_acts.detach().cpu().numpy().tolist(),
            "top_features_per_token": top_features_per_token.detach().cpu().numpy().tolist(),
            "top_acts_per_token": top_acts_per_token.detach().cpu().numpy().tolist(),
            "top_features_global": global_feature_info,
            "num_features": sae_acts.shape[-1],
        }

    def analyze_prompt_sequential(
        self,
        prompt: str,
        layers: list[int] = None,
        top_k: int = 10,
    ) -> dict:
        """
        Analyze a prompt using sequential loading (LLM and SAE never coexist).

        This is a convenience method that combines gather_and_cache_residuals()
        and encode_cached_residuals() for a complete analysis workflow.

        Args:
            prompt: The text prompt to analyze
            layers: List of layers to analyze (defaults to all SAE layers)
            top_k: Number of top features to return per token

        Returns:
            Dictionary with tokens and per-layer activation data
        """
        if layers is None:
            layers = self.sae_layers

        # Step 1: Gather residuals and unload LLM
        cache_result = self.gather_and_cache_residuals(prompt, unload_llm_after=True)
        cache_key = cache_result["cache_key"]
        str_tokens = cache_result["tokens"]

        # Step 2: Encode with SAE for each layer
        layers_data = {}
        for layer_idx in layers:
            layer_result = self.encode_cached_residuals(
                cache_key,
                layer_idx,
                top_k=top_k,
                unload_sae_after=True,  # Unload each SAE after use
            )
            layers_data[layer_idx] = {
                "sae_acts": layer_result["sae_acts"],
                "top_features_per_token": layer_result["top_features_per_token"],
                "top_acts_per_token": layer_result["top_acts_per_token"],
                "top_features_global": layer_result["top_features_global"],
                "num_features": layer_result["num_features"],
            }

        # Clean up cache
        if cache_key in _residual_cache:
            del _residual_cache[cache_key]

        return {
            "tokens": str_tokens,
            "num_tokens": len(str_tokens),
            "layers": layers_data,
            "available_layers": self.sae_layers,
            "sae_config": {
                "width": self.sae_width,
                "l0": self.sae_l0,
            },
            "sequential_mode": True,
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

    def analyze_layer(self, prompt: str, layer: int, top_k: int = 10, unload_others: bool = True) -> dict:
        """
        Analyze a specific layer for a prompt. Loads SAE weights on-demand.
        Uses cached residual activations if available.

        Args:
            prompt: The text prompt to analyze
            layer: The SAE layer index to analyze
            top_k: Number of top features to return per token
            unload_others: If True, unload other SAEs to save memory (default True)

        Returns:
            Dictionary with SAE activation data for this layer
        """
        if layer not in self.sae_layers:
            raise ValueError(f"Layer {layer} not in available layers: {self.sae_layers}")

        # Unload other SAEs to prevent memory accumulation
        if unload_others:
            layers_to_unload = [l for l in list(self._saes.keys()) if l != layer]
            for l in layers_to_unload:
                self.unload_sae(l)

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

    def get_feature_activations(self, prompt: str, feature_id: int, layer: int = None, unload_others: bool = True) -> dict:
        """
        Get activation pattern for a specific feature across all tokens.

        Args:
            prompt: The text prompt
            feature_id: The SAE feature index
            layer: The SAE layer (defaults to first available layer)
            unload_others: If True, unload other SAEs to save memory (default True)

        Returns:
            Dictionary with per-token activations for the feature
        """
        if layer is None:
            layer = self.sae_layers[0]

        # Unload other SAEs to prevent memory accumulation
        if unload_others:
            layers_to_unload = [l for l in list(self._saes.keys()) if l != layer]
            for l in layers_to_unload:
                self.unload_sae(l)

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
