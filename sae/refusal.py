"""
Refusal Detection Mixin

Methods for detecting refusal patterns and ranking features.
"""

import torch
from .cache import _residual_cache


# Default refusal phrases to detect
REFUSAL_PHRASES = [
    "I can't", "I cannot", "I'm sorry", "I apologize",
    "I'm not able", "I am not able", "I won't", "I will not",
    "I'm unable", "I am unable", "As an AI",
    "I must decline", "I cannot assist", "I can't help",
    "against my guidelines", "harmful", "unethical",
    "I'm not going to", "I refuse",
]


class RefusalMixin:
    """Mixin providing refusal detection methods for SAEModelManager."""

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
            refusal_phrases = REFUSAL_PHRASES

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

    def detect_refusal_lazy(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        refusal_phrases: list[str] = None,
    ) -> dict:
        """
        Generate a response and detect refusal patterns WITHOUT loading SAE.
        Caches residuals for later layer-by-layer analysis.

        Returns:
            Dictionary with refusal detection info and available layers
        """
        if refusal_phrases is None:
            refusal_phrases = REFUSAL_PHRASES

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
                char_count = 0
                for i, tok in enumerate(generated_tokens):
                    char_count += len(tok)
                    if char_count >= pos:
                        refusal_positions.append(prompt_len + i)
                        break

        # Cache residual activations using a unique key for this generation
        cache_key = hash(f"refusal_{prompt}_{max_new_tokens}")
        residuals = self.gather_residual_activations(outputs)
        _residual_cache[cache_key] = {
            "residuals": residuals,
            "tokens": full_tokens,
            "inputs": outputs,
            "prompt_len": prompt_len,
            "refusal_positions": refusal_positions,
        }

        return {
            "prompt": prompt,
            "generated_text": generated_text,
            "refusal_detected": refusal_detected,
            "refusal_phrases_found": phrases_found,
            "generated_tokens": generated_tokens,
            "refusal_token_positions": refusal_positions,
            "prompt_len": prompt_len,
            "available_layers": self.sae_layers,
            "cache_key": str(cache_key),  # String to avoid JS integer precision loss
        }

    def detect_refusal_layer(
        self,
        cache_key: int | str,
        layer: int,
        unload_others: bool = True,
    ) -> dict:
        """
        Analyze refusal-correlated features for a SINGLE layer.
        Uses cached residuals from detect_refusal_lazy.

        Args:
            cache_key: Cache key from detect_refusal_lazy (string or int)
            layer: The SAE layer to analyze
            unload_others: If True, unload other SAEs to save memory (default True)

        Returns:
            Dictionary with refusal-correlated features for this layer
        """
        # Convert string cache key back to int (handles JS integer precision loss)
        if isinstance(cache_key, str):
            cache_key = int(cache_key)

        if layer not in self.sae_layers:
            raise ValueError(f"Layer {layer} not in available layers: {self.sae_layers}")

        if cache_key not in _residual_cache:
            raise ValueError("Cached residuals not found. Run detect_refusal_lazy first.")

        # Unload other SAEs to prevent memory accumulation
        if unload_others:
            layers_to_unload = [l for l in list(self._saes.keys()) if l != layer]
            for l in layers_to_unload:
                self.unload_sae(l)

        cached = _residual_cache[cache_key]
        residuals = cached["residuals"][layer]
        prompt_len = cached["prompt_len"]
        refusal_positions = cached["refusal_positions"]

        # Load SAE for this layer
        sae = self.get_sae(layer)
        sae_acts = sae.encode(residuals.to(torch.float32))

        # Compare refusal positions to overall mean
        gen_acts = sae_acts[prompt_len:]  # Only generated tokens
        mean_gen_acts = gen_acts.mean(dim=0)

        # If we have refusal positions, compare them
        if refusal_positions:
            refusal_indices = [p - prompt_len for p in refusal_positions if p - prompt_len < len(gen_acts)]
            if refusal_indices:
                refusal_acts = gen_acts[refusal_indices].mean(dim=0)
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
                    "neuronpedia_url": self.get_neuronpedia_url(feat_idx, layer),
                })

        return {
            "layer": layer,
            "refusal_correlated_features": refusal_features[:20],
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

    def rank_features_single_category(
        self,
        prompts: list[str],
        category: str = "harmful",
        top_k: int = 100,
    ) -> dict:
        """
        Rank features by their mean activation strength on a single category of prompts.

        Args:
            prompts: List of prompt strings
            category: Label for the category ("harmful" or "harmless")
            top_k: Number of top features to return per layer

        Returns:
            Dictionary with ranked features per layer
        """
        # Initialize accumulators per layer
        activation_sums = {layer: None for layer in self.sae_layers}
        num_prompts = len(prompts)

        for prompt in prompts:
            # Tokenize
            inputs = self.tokenizer.encode(
                prompt, return_tensors="pt", add_special_tokens=True
            ).to(self.device)

            # Get activations
            residuals = self.gather_residual_activations(inputs)

            for layer_idx in self.sae_layers:
                sae = self.get_sae(layer_idx)
                acts = sae.encode(residuals[layer_idx].to(torch.float32))
                mean_act = acts[1:].mean(dim=0)  # Skip BOS

                # Initialize accumulator on first prompt
                if activation_sums[layer_idx] is None:
                    activation_sums[layer_idx] = torch.zeros_like(mean_act)

                activation_sums[layer_idx] += mean_act

        # Compute rankings
        layers_data = {}
        for layer_idx in self.sae_layers:
            if activation_sums[layer_idx] is None:
                layers_data[layer_idx] = {"ranked_features": []}
                continue

            mean_activation = activation_sums[layer_idx] / num_prompts

            # Get top features by mean activation
            top_scores, top_indices = mean_activation.topk(top_k)

            ranked_features = []
            for score, feat_idx in zip(top_scores.tolist(), top_indices.tolist()):
                ranked_features.append({
                    "feature_id": feat_idx,
                    "mean_activation": round(score, 4),
                    "neuronpedia_url": self.get_neuronpedia_url(feat_idx, layer_idx),
                })

            layers_data[layer_idx] = {"ranked_features": ranked_features}

        return {
            "num_prompts": num_prompts,
            "category": category,
            "layers": layers_data,
            "available_layers": self.sae_layers,
        }

    def rank_features_lazy(
        self,
        prompt_pairs: list[dict] = None,
        prompts: list[str] = None,
        category: str = None,
    ) -> dict:
        """
        Tokenize and cache residuals for batch ranking WITHOUT loading SAE.

        Supports both pair mode and single-category mode.

        Args:
            prompt_pairs: List of {"harmful": str, "benign": str} dicts (pair mode)
            prompts: List of prompt strings (single-category mode)
            category: "harmful" or "harmless" (single-category mode)

        Returns:
            Dictionary with cache key and available layers
        """
        is_pair_mode = prompt_pairs is not None
        cache_entries = []

        if is_pair_mode:
            for pair in prompt_pairs:
                harmful = pair.get("harmful", "")
                benign = pair.get("benign", "")

                # Tokenize and cache harmful
                cache_key_h = hash(harmful)
                if cache_key_h not in _residual_cache:
                    inputs_h = self.tokenizer.encode(
                        harmful, return_tensors="pt", add_special_tokens=True
                    ).to(self.device)
                    residuals_h = self.gather_residual_activations(inputs_h)
                    tokens_h = self.tokenizer.convert_ids_to_tokens(inputs_h[0].tolist())
                    tokens_h = [t.replace("▁", " ").replace("<0x0A>", "\\n") for t in tokens_h]
                    _residual_cache[cache_key_h] = {
                        "residuals": residuals_h,
                        "tokens": tokens_h,
                        "inputs": inputs_h,
                    }

                # Tokenize and cache benign
                cache_key_b = hash(benign)
                if cache_key_b not in _residual_cache:
                    inputs_b = self.tokenizer.encode(
                        benign, return_tensors="pt", add_special_tokens=True
                    ).to(self.device)
                    residuals_b = self.gather_residual_activations(inputs_b)
                    tokens_b = self.tokenizer.convert_ids_to_tokens(inputs_b[0].tolist())
                    tokens_b = [t.replace("▁", " ").replace("<0x0A>", "\\n") for t in tokens_b]
                    _residual_cache[cache_key_b] = {
                        "residuals": residuals_b,
                        "tokens": tokens_b,
                        "inputs": inputs_b,
                    }

                cache_entries.append({"harmful_key": cache_key_h, "benign_key": cache_key_b})

            # Create a master cache key for this ranking session
            master_key = hash(f"rank_pairs_{len(prompt_pairs)}_{hash(str(prompt_pairs[:3]))}")
            _residual_cache[master_key] = {
                "mode": "pairs",
                "entries": cache_entries,
                "num_pairs": len(prompt_pairs),
            }

            return {
                "cache_key": str(master_key),  # String to avoid JS integer precision loss
                "num_prompt_pairs": len(prompt_pairs),
                "available_layers": self.sae_layers,
            }

        else:
            # Single-category mode
            for prompt in prompts:
                cache_key = hash(prompt)
                if cache_key not in _residual_cache:
                    inputs = self.tokenizer.encode(
                        prompt, return_tensors="pt", add_special_tokens=True
                    ).to(self.device)
                    residuals = self.gather_residual_activations(inputs)
                    tokens = self.tokenizer.convert_ids_to_tokens(inputs[0].tolist())
                    tokens = [t.replace("▁", " ").replace("<0x0A>", "\\n") for t in tokens]
                    _residual_cache[cache_key] = {
                        "residuals": residuals,
                        "tokens": tokens,
                        "inputs": inputs,
                    }
                cache_entries.append(cache_key)

            # Create a master cache key for this ranking session
            master_key = hash(f"rank_single_{category}_{len(prompts)}_{hash(str(prompts[:3]))}")
            _residual_cache[master_key] = {
                "mode": "single",
                "category": category,
                "entries": cache_entries,
                "num_prompts": len(prompts),
            }

            return {
                "cache_key": str(master_key),  # String to avoid JS integer precision loss
                "num_prompts": len(prompts),
                "category": category,
                "available_layers": self.sae_layers,
            }

    def rank_features_layer(
        self,
        cache_key: int | str,
        layer: int,
        top_k: int = 100,
        unload_others: bool = True,
    ) -> dict:
        """
        Rank features for a SINGLE layer using cached residuals.

        Args:
            cache_key: Cache key from rank_features_lazy (string or int)
            layer: The SAE layer to analyze
            top_k: Number of top features to return
            unload_others: If True, unload other SAEs to save memory (default True)

        Returns:
            Dictionary with ranked features for this layer
        """
        # Convert string cache key back to int (handles JS integer precision loss)
        if isinstance(cache_key, str):
            cache_key = int(cache_key)

        if layer not in self.sae_layers:
            raise ValueError(f"Layer {layer} not in available layers: {self.sae_layers}")

        if cache_key not in _residual_cache:
            raise ValueError("Cached data not found. Run rank_features_lazy first.")

        # Unload other SAEs to prevent memory accumulation
        if unload_others:
            layers_to_unload = [l for l in list(self._saes.keys()) if l != layer]
            for l in layers_to_unload:
                self.unload_sae(l)

        cached = _residual_cache[cache_key]
        mode = cached.get("mode")

        # Load SAE for this layer
        sae = self.get_sae(layer)

        if mode == "pairs":
            # Pair mode: compute differential scores
            entries = cached["entries"]
            num_pairs = cached["num_pairs"]

            harmful_sum = None
            benign_sum = None
            diff_positive_count = None

            for entry in entries:
                residuals_h = _residual_cache[entry["harmful_key"]]["residuals"][layer]
                residuals_b = _residual_cache[entry["benign_key"]]["residuals"][layer]

                acts_h = sae.encode(residuals_h.to(device=self.device, dtype=torch.float32))
                acts_b = sae.encode(residuals_b.to(device=self.device, dtype=torch.float32))

                mean_h = acts_h[1:].mean(dim=0)  # Skip BOS
                mean_b = acts_b[1:].mean(dim=0)

                if harmful_sum is None:
                    harmful_sum = torch.zeros_like(mean_h)
                    benign_sum = torch.zeros_like(mean_b)
                    diff_positive_count = torch.zeros_like(mean_h)

                harmful_sum += mean_h
                benign_sum += mean_b
                diff_positive_count += (mean_h > mean_b).float()

            # Compute rankings
            mean_harmful = harmful_sum / num_pairs
            mean_benign = benign_sum / num_pairs
            consistency = diff_positive_count / num_pairs

            mean_diff = mean_harmful - mean_benign
            differential_score = consistency * mean_diff.abs()

            top_scores, top_indices = differential_score.topk(top_k)

            ranked_features = []
            for score, feat_idx in zip(top_scores.tolist(), top_indices.tolist()):
                ranked_features.append({
                    "feature_id": feat_idx,
                    "consistency_score": round(consistency[feat_idx].item(), 4),
                    "mean_harmful_activation": round(mean_harmful[feat_idx].item(), 4),
                    "mean_benign_activation": round(mean_benign[feat_idx].item(), 4),
                    "differential_score": round(score, 4),
                    "neuronpedia_url": self.get_neuronpedia_url(feat_idx, layer),
                })

            return {
                "layer": layer,
                "ranked_features": ranked_features,
            }

        else:
            # Single-category mode: compute mean activations
            entries = cached["entries"]
            num_prompts = cached["num_prompts"]

            activation_sum = None

            for prompt_key in entries:
                residuals = _residual_cache[prompt_key]["residuals"][layer]
                acts = sae.encode(residuals.to(device=self.device, dtype=torch.float32))
                mean_act = acts[1:].mean(dim=0)

                if activation_sum is None:
                    activation_sum = torch.zeros_like(mean_act)

                activation_sum += mean_act

            mean_activation = activation_sum / num_prompts
            top_scores, top_indices = mean_activation.topk(top_k)

            ranked_features = []
            for score, feat_idx in zip(top_scores.tolist(), top_indices.tolist()):
                ranked_features.append({
                    "feature_id": feat_idx,
                    "mean_activation": round(score, 4),
                    "neuronpedia_url": self.get_neuronpedia_url(feat_idx, layer),
                })

            return {
                "layer": layer,
                "ranked_features": ranked_features,
            }
