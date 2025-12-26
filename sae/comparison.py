"""
Comparison Mixin

Methods for comparing activations between prompts.
"""

import torch
from .cache import _residual_cache


class ComparisonMixin:
    """Mixin providing prompt comparison methods for SAEModelManager."""

    def compare_prompts_sequential(
        self,
        prompt_a: str,
        prompt_b: str,
        layers: list[int] = None,
        top_k: int = 50,
        threshold: float = 0.1,
    ) -> dict:
        """
        Compare two prompts using sequential loading (LLM and SAE never coexist).

        Args:
            prompt_a: First prompt (typically harmful/refused)
            prompt_b: Second prompt (typically benign/accepted)
            layers: List of layers to analyze (defaults to all SAE layers)
            top_k: Number of top differential features per layer
            threshold: Minimum activation to consider

        Returns:
            Dictionary with differential features per layer
        """
        if layers is None:
            layers = self.sae_layers

        # Step 1: Gather residuals for both prompts, then unload LLM
        result_a = self.gather_and_cache_residuals(prompt_a, unload_llm_after=False)
        result_b = self.gather_and_cache_residuals(prompt_b, unload_llm_after=True)

        cache_key_a = result_a["cache_key"]
        cache_key_b = result_b["cache_key"]
        tokens_a = result_a["tokens"]
        tokens_b = result_b["tokens"]

        # Step 2: Compare activations for each layer
        layers_data = {}
        for layer_idx in layers:
            # Get cached residuals
            residuals_a = _residual_cache[int(cache_key_a)]["residuals"][layer_idx].to(self.device)
            residuals_b = _residual_cache[int(cache_key_b)]["residuals"][layer_idx].to(self.device)

            # Load SAE (unloading others)
            layers_to_unload = [l for l in list(self._saes.keys()) if l != layer_idx]
            for l in layers_to_unload:
                self.unload_sae(l)

            sae = self.get_sae(layer_idx)

            # Encode with SAE
            sae_acts_a = sae.encode(residuals_a.to(torch.float32))
            sae_acts_b = sae.encode(residuals_b.to(torch.float32))

            # Compute mean activations (skip BOS)
            mean_acts_a = sae_acts_a[1:].mean(dim=0)
            mean_acts_b = sae_acts_b[1:].mean(dim=0)

            # Compute differential
            diff = mean_acts_a - mean_acts_b
            epsilon = 1e-6
            ratio = mean_acts_a / (mean_acts_b + epsilon)

            # Get features above threshold
            mask = (mean_acts_a > threshold) | (mean_acts_b > threshold)
            valid_indices = torch.where(mask)[0]

            differential_features = []
            token_acts_a = {}
            token_acts_b = {}

            if len(valid_indices) > 0:
                valid_diffs = diff[valid_indices]
                valid_acts_a = mean_acts_a[valid_indices]
                valid_acts_b = mean_acts_b[valid_indices]
                valid_ratios = ratio[valid_indices]

                sorted_indices = torch.argsort(valid_diffs.abs(), descending=True)[:top_k]

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

                top_feature_ids = [f["feature_id"] for f in differential_features]
                for feat_id in top_feature_ids:
                    token_acts_a[feat_id] = [round(v, 4) for v in sae_acts_a[:, feat_id].tolist()]
                    token_acts_b[feat_id] = [round(v, 4) for v in sae_acts_b[:, feat_id].tolist()]

            layers_data[layer_idx] = {
                "differential_features": differential_features,
                "token_activations_a": token_acts_a,
                "token_activations_b": token_acts_b,
            }

            # Clean up GPU tensors
            del residuals_a, residuals_b

            # Unload SAE after each layer
            self.unload_sae(layer_idx)

        # Clean up caches
        cache_key_a_int = int(cache_key_a)
        cache_key_b_int = int(cache_key_b)
        if cache_key_a_int in _residual_cache:
            del _residual_cache[cache_key_a_int]
        if cache_key_b_int in _residual_cache:
            del _residual_cache[cache_key_b_int]

        return {
            "prompt_a": prompt_a,
            "prompt_b": prompt_b,
            "tokens_a": tokens_a,
            "tokens_b": tokens_b,
            "layers": layers_data,
            "available_layers": self.sae_layers,
            "sequential_mode": True,
        }

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

    def compare_prompts_lazy(
        self,
        prompt_a: str,
        prompt_b: str,
    ) -> dict:
        """
        Initial comparison that tokenizes both prompts and caches residuals.
        Does NOT load any SAE weights.

        Returns:
            Dictionary with tokens for both prompts and available layers
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

        # Cache residual activations for both prompts
        cache_key_a = hash(prompt_a)
        cache_key_b = hash(prompt_b)

        if cache_key_a not in _residual_cache:
            residuals_a = self.gather_residual_activations(inputs_a)
            _residual_cache[cache_key_a] = {
                "residuals": residuals_a,
                "tokens": tokens_a,
                "inputs": inputs_a,
            }

        if cache_key_b not in _residual_cache:
            residuals_b = self.gather_residual_activations(inputs_b)
            _residual_cache[cache_key_b] = {
                "residuals": residuals_b,
                "tokens": tokens_b,
                "inputs": inputs_b,
            }

        return {
            "prompt_a": prompt_a,
            "prompt_b": prompt_b,
            "tokens_a": tokens_a,
            "tokens_b": tokens_b,
            "available_layers": self.sae_layers,
        }

    def compare_prompts_layer(
        self,
        prompt_a: str,
        prompt_b: str,
        layer: int,
        top_k: int = 50,
        threshold: float = 0.1,
        unload_others: bool = True
    ) -> dict:
        """
        Compare activations between two prompts for a SINGLE layer.
        Uses cached residual activations if available.

        Args:
            prompt_a: First prompt (typically harmful/refused)
            prompt_b: Second prompt (typically benign/accepted)
            layer: The SAE layer to analyze
            top_k: Number of top differential features
            threshold: Minimum activation to consider
            unload_others: If True, unload other SAEs to save memory (default True)

        Returns:
            Dictionary with differential features for this layer
        """
        if layer not in self.sae_layers:
            raise ValueError(f"Layer {layer} not in available layers: {self.sae_layers}")

        # Unload other SAEs to prevent memory accumulation
        if unload_others:
            layers_to_unload = [l for l in list(self._saes.keys()) if l != layer]
            for l in layers_to_unload:
                self.unload_sae(l)

        cache_key_a = hash(prompt_a)
        cache_key_b = hash(prompt_b)

        # Get cached residuals or compute them
        if cache_key_a in _residual_cache:
            residuals_a = _residual_cache[cache_key_a]["residuals"][layer]
            tokens_a = _residual_cache[cache_key_a]["tokens"]
        else:
            inputs_a = self.tokenizer.encode(
                prompt_a, return_tensors="pt", add_special_tokens=True
            ).to(self.device)
            tokens_a = self.tokenizer.convert_ids_to_tokens(inputs_a[0].tolist())
            tokens_a = [t.replace("▁", " ").replace("<0x0A>", "\\n") for t in tokens_a]
            all_residuals_a = self.gather_residual_activations(inputs_a)
            _residual_cache[cache_key_a] = {
                "residuals": all_residuals_a,
                "tokens": tokens_a,
                "inputs": inputs_a,
            }
            residuals_a = all_residuals_a[layer]

        if cache_key_b in _residual_cache:
            residuals_b = _residual_cache[cache_key_b]["residuals"][layer]
            tokens_b = _residual_cache[cache_key_b]["tokens"]
        else:
            inputs_b = self.tokenizer.encode(
                prompt_b, return_tensors="pt", add_special_tokens=True
            ).to(self.device)
            tokens_b = self.tokenizer.convert_ids_to_tokens(inputs_b[0].tolist())
            tokens_b = [t.replace("▁", " ").replace("<0x0A>", "\\n") for t in tokens_b]
            all_residuals_b = self.gather_residual_activations(inputs_b)
            _residual_cache[cache_key_b] = {
                "residuals": all_residuals_b,
                "tokens": tokens_b,
                "inputs": inputs_b,
            }
            residuals_b = all_residuals_b[layer]

        # Load SAE for this layer
        sae = self.get_sae(layer)

        # Encode with SAE
        sae_acts_a = sae.encode(residuals_a.to(torch.float32))
        sae_acts_b = sae.encode(residuals_b.to(torch.float32))

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

        differential_features = []
        token_acts_a = {}
        token_acts_b = {}

        if len(valid_indices) > 0:
            # Get differential values for valid features
            valid_diffs = diff[valid_indices]
            valid_acts_a = mean_acts_a[valid_indices]
            valid_acts_b = mean_acts_b[valid_indices]
            valid_ratios = ratio[valid_indices]

            # Sort by absolute differential (most different first)
            sorted_indices = torch.argsort(valid_diffs.abs(), descending=True)[:top_k]

            for idx in sorted_indices:
                feat_idx = valid_indices[idx].item()
                differential_features.append({
                    "feature_id": feat_idx,
                    "mean_diff": round(valid_diffs[idx].item(), 4),
                    "activation_a": round(valid_acts_a[idx].item(), 4),
                    "activation_b": round(valid_acts_b[idx].item(), 4),
                    "ratio": round(valid_ratios[idx].item(), 4),
                    "neuronpedia_url": self.get_neuronpedia_url(feat_idx, layer),
                })

            # Store per-token activations for top features
            top_feature_ids = [f["feature_id"] for f in differential_features]
            for feat_id in top_feature_ids:
                token_acts_a[feat_id] = [round(v, 4) for v in sae_acts_a[:, feat_id].tolist()]
                token_acts_b[feat_id] = [round(v, 4) for v in sae_acts_b[:, feat_id].tolist()]

        return {
            "layer": layer,
            "tokens_a": tokens_a,
            "tokens_b": tokens_b,
            "differential_features": differential_features,
            "token_activations_a": token_acts_a,
            "token_activations_b": token_acts_b,
        }
