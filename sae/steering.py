"""
Steering Mixin

Methods for generating text with feature steering applied.
"""

import os
import torch
import torch.nn as nn


class SteeringMixin:
    """Mixin providing steering methods for SAEModelManager."""

    def generate_with_steering(
        self,
        prompt: str,
        steering_features: list[dict],
        max_new_tokens: int = 256,
        steering_layer_offset: int = -5,
        sae_layer: int = None,
        normalization: str = None,
        norm_clamp_factor: float = 1.5,
        unit_normalize: bool = False,
        skip_baseline: bool = False,
    ) -> dict:
        """
        Generate text with feature steering applied.

        Args:
            prompt: The input prompt
            steering_features: List of {"feature_id": int, "coefficient": float, "layer": int (optional)}
            max_new_tokens: Maximum tokens to generate
            steering_layer_offset: Layer offset from SAE layer for steering
            sae_layer: Default SAE layer for features without explicit layer
            normalization: Post-steering normalization mode:
                - None: No post-steering normalization
                - "preserve_norm": Rescale output to maintain original norm after steering
                - "clamp": Clamp norm change to norm_clamp_factor
            norm_clamp_factor: For "clamp" mode, max allowed norm change ratio (default 1.5)
            unit_normalize: If True, normalize decoder vectors to unit norm before applying.
                           Can be combined with any normalization mode.
            skip_baseline: If True, skip generating baseline (non-steered) output.

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

        # Generate without steering first (unless skipped)
        original_output = None
        if not skip_baseline:
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

            def make_steering_hook(sae_ref, feats, norm_mode, clamp_factor, unit_norm):
                def steering_hook(mod, inputs, outputs):
                    output = outputs[0]

                    # Handle KV caching - different logic for first vs cached passes
                    if output.shape[1] == 1:
                        # Single token (cached generation)
                        original_norm = torch.norm(output, dim=-1, keepdim=True)

                        for sf in feats:
                            feat_idx = sf["feature_id"]
                            coeff = sf["coefficient"]
                            decoder_vec = sae_ref.w_dec[feat_idx]

                            # Optionally normalize decoder vector to unit norm
                            if unit_norm:
                                decoder_vec = decoder_vec / (torch.norm(decoder_vec) + 1e-8)

                            output += coeff * original_norm * decoder_vec

                        # Apply post-steering normalization
                        if norm_mode == "preserve_norm":
                            new_norm = torch.norm(output, dim=-1, keepdim=True)
                            output *= original_norm / (new_norm + 1e-8)
                        elif norm_mode == "clamp":
                            new_norm = torch.norm(output, dim=-1, keepdim=True)
                            norm_ratio = new_norm / (original_norm + 1e-8)
                            if norm_ratio > clamp_factor:
                                output *= clamp_factor / norm_ratio
                            elif norm_ratio < 1.0 / clamp_factor:
                                output *= (1.0 / clamp_factor) / norm_ratio
                    else:
                        # Full sequence (first pass)
                        original_norms = torch.norm(output[0, 1:], dim=-1, keepdim=True)

                        for sf in feats:
                            feat_idx = sf["feature_id"]
                            coeff = sf["coefficient"]
                            decoder_vec = sae_ref.w_dec[feat_idx]

                            # Optionally normalize decoder vector to unit norm
                            if unit_norm:
                                decoder_vec = decoder_vec / (torch.norm(decoder_vec) + 1e-8)

                            output[0, 1:] += coeff * original_norms * decoder_vec

                        # Apply post-steering normalization
                        if norm_mode == "preserve_norm":
                            new_norms = torch.norm(output[0, 1:], dim=-1, keepdim=True)
                            output[0, 1:] *= original_norms / (new_norms + 1e-8)
                        elif norm_mode == "clamp":
                            new_norms = torch.norm(output[0, 1:], dim=-1, keepdim=True)
                            norm_ratios = new_norms / (original_norms + 1e-8)
                            scale = torch.ones_like(norm_ratios)
                            scale = torch.where(norm_ratios > clamp_factor, clamp_factor / norm_ratios, scale)
                            scale = torch.where(norm_ratios < 1.0 / clamp_factor, (1.0 / clamp_factor) / norm_ratios, scale)
                            output[0, 1:] *= scale

                    return outputs
                return steering_hook

            handle = self.layers[steering_target_layer].register_forward_hook(
                make_steering_hook(sae, layer_features, normalization, norm_clamp_factor, unit_normalize)
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
            "normalization": normalization,
            "unit_normalize": unit_normalize,
        }

    def apply_steering_to_weights(
        self,
        features: list[dict],
        output_path: str,
        scale_factor: float = 1.0,
    ) -> dict:
        """
        Apply steering vectors permanently to model weights using abliteration-style
        weight surgery. Works with any architecture (no bias required).

        This modifies the MLP down_proj weights to amplify or suppress feature directions:
        - Positive coefficient: amplify the feature direction
        - Negative coefficient: suppress/abliterate the feature direction

        The modification formula:
            W_new = W + scale * (d @ d.T) @ W.T / ||d||²

        Where d is the decoder vector (feature direction in residual stream space).

        Args:
            features: List of {layer, feature_id, coefficient} dicts
            output_path: Path to save the modified model
            scale_factor: Additional scaling for the steering vectors (default 1.0)

        Returns:
            Dictionary with status and details
        """
        import json
        from datetime import datetime

        # Ensure model is loaded
        _ = self.model

        # Group features by layer
        layer_features: dict[int, list[dict]] = {}
        for feat in features:
            layer = feat.get("layer", self.sae_layers[0] if self.sae_layers else 0)
            if layer not in layer_features:
                layer_features[layer] = []
            layer_features[layer].append(feat)

        modifications = []

        with torch.no_grad():
            for layer_idx, feats in layer_features.items():
                # Load SAE for this layer
                sae = self.get_sae(layer_idx)

                # Get the layer's MLP down projection
                layer = self.layers[layer_idx]

                # Find the MLP down_proj - handle different architectures
                down_proj = None
                down_proj_path = None

                if hasattr(layer, 'mlp'):
                    mlp = layer.mlp
                    if hasattr(mlp, 'down_proj'):
                        down_proj = mlp.down_proj
                        down_proj_path = f"layers.{layer_idx}.mlp.down_proj"
                elif hasattr(layer, 'feed_forward'):
                    mlp = layer.feed_forward
                    if hasattr(mlp, 'w2'):
                        down_proj = mlp.w2
                        down_proj_path = f"layers.{layer_idx}.feed_forward.w2"

                if down_proj is None:
                    # Try to find any linear layer we can modify
                    for name, module in layer.named_modules():
                        if 'down' in name.lower() and hasattr(module, 'weight'):
                            down_proj = module
                            down_proj_path = f"layers.{layer_idx}.{name}"
                            break

                if down_proj is None:
                    raise ValueError(f"Could not find MLP down projection for layer {layer_idx}")

                weight_dtype = down_proj.weight.dtype
                weight_device = down_proj.weight.device

                # Apply abliteration-style weight surgery for each feature
                for feat in feats:
                    feat_idx = feat["feature_id"]
                    coeff = feat["coefficient"] * scale_factor

                    # Get decoder vector (feature direction in residual stream)
                    decoder_vec = sae.w_dec[feat_idx].to(device=weight_device, dtype=weight_dtype)

                    # Normalize the direction
                    direction_norm = torch.norm(decoder_vec)
                    if direction_norm < 1e-8:
                        print(f"Warning: Feature {feat_idx} has near-zero norm, skipping")
                        continue

                    # Compute projection matrix component: d @ d.T / ||d||²
                    # down_proj.weight shape: (hidden_dim, intermediate_dim)
                    # decoder_vec shape: (hidden_dim,)
                    #
                    # We want to modify how much the output projects onto this direction
                    # W_new = W + coeff * (d ⊗ d) @ W / ||d||²
                    # But more directly: modify the output projection along d

                    # Project each column of W onto d, then scale that projection
                    # proj = (W.T @ d) gives how much each intermediate dim contributes to d
                    # W_new = W + coeff * d.unsqueeze(1) @ proj.unsqueeze(0) / ||d||²

                    proj = down_proj.weight.T @ decoder_vec  # (intermediate_dim,)
                    delta = coeff * torch.outer(decoder_vec, proj) / (direction_norm ** 2)

                    down_proj.weight.data += delta

                    action = "amplified" if coeff > 0 else "suppressed"
                    print(f"Layer {layer_idx}: {action} feature {feat_idx} (coeff={coeff:.3f})")

                modifications.append({
                    "layer": layer_idx,
                    "features": len(feats),
                    "target_module": down_proj_path,
                    "method": "abliteration",
                })

        # Save the modified model
        print(f"Saving modified model to {output_path}...")
        os.makedirs(output_path, exist_ok=True)

        # Save model and tokenizer
        self._model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        # Save steering configuration for reproducibility
        steering_config = {
            "created_at": datetime.now().isoformat(),
            "source_model": self.model_path,
            "sae_repo": self.sae_repo,
            "sae_width": self.sae_width,
            "sae_l0": self.sae_l0,
            "scale_factor": scale_factor,
            "method": "abliteration",
            "features": features,
            "modifications": modifications,
        }
        config_path = os.path.join(output_path, "steering_config.json")
        with open(config_path, "w") as f:
            json.dump(steering_config, f, indent=2)
        print(f"Saved steering configuration to {config_path}")

        return {
            "success": True,
            "output_path": output_path,
            "config_path": config_path,
            "modifications": modifications,
            "total_features": sum(len(lf) for lf in layer_features.values()),
        }
