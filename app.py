"""
Flask LM Feature Studio - Batch Ranking Mode

A local web app for exploring SAE features on Gemma models,
focused on batch feature ranking across prompt pairs.
"""

import os
import json
import csv
import io
from datetime import datetime
from flask import Flask, render_template, request, jsonify, Response
from sae import get_manager, initialize_models

app = Flask(__name__)

# Configuration
PRELOAD_MODELS = os.environ.get("PRELOAD_MODELS", "0") == "1"


def use_sequential_mode() -> bool:
    """
    Check if sequential loading mode should be used.

    Sequential mode loads LLM and SAE separately (never simultaneously)
    to reduce peak GPU memory usage. Enabled automatically for 12b models.
    """
    manager = get_manager()
    return manager.base_model == "12b"


@app.route("/")
def index():
    """Serve the main UI."""
    return render_template("index.html")


@app.route("/api/neuronpedia/<int:layer>/<int:feature_id>", methods=["GET"])
def get_neuronpedia_data(layer: int, feature_id: int):
    """
    Fetch feature data from Neuronpedia.

    Returns explanations, lists, positive/negative logits, and top activations
    for the specified feature.
    """
    try:
        manager = get_manager()
        result = manager.fetch_neuronpedia_data(feature_id, layer)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/steer", methods=["POST"])
def steer_generation():
    """
    Generate text with feature steering applied.

    Request JSON:
        {
            "prompt": "Tell me a fun fact",
            "steering": [{"feature_id": 123, "coefficient": 0.25, "layer": 12}],
            "max_tokens": 80,
            "normalization": "preserve_norm",
            "unit_normalize": true,
            "skip_baseline": false
        }
    """
    data = request.get_json()
    prompt = data.get("prompt", "")
    steering = data.get("steering", [])
    max_tokens = data.get("max_tokens", 256)
    normalization = data.get("normalization", None)
    norm_clamp_factor = data.get("norm_clamp_factor", 1.5)
    unit_normalize = data.get("unit_normalize", False)
    skip_baseline = data.get("skip_baseline", False)

    if not prompt.strip():
        return jsonify({"error": "Prompt cannot be empty"}), 400

    valid_modes = [None, "preserve_norm", "clamp"]
    if normalization not in valid_modes:
        return jsonify({"error": f"Invalid normalization mode. Use one of: {valid_modes}"}), 400

    for sf in steering:
        if "feature_id" not in sf or "coefficient" not in sf:
            return jsonify({"error": "Each steering entry needs feature_id and coefficient"}), 400

    try:
        manager = get_manager()
        result = manager.generate_with_steering(
            prompt=prompt,
            steering_features=steering,
            max_new_tokens=max_tokens,
            normalization=normalization,
            norm_clamp_factor=norm_clamp_factor,
            unit_normalize=unit_normalize,
            skip_baseline=skip_baseline,
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/config", methods=["GET"])
def get_config():
    """Get current SAE configuration."""
    from sae import get_neuronpedia_layers
    manager = get_manager()
    return jsonify({
        "model_path": manager.model_path,
        "sae_repo": manager.sae_repo,
        "sae_layers": manager.sae_layers,
        "sae_width": manager.sae_width,
        "sae_l0": manager.sae_l0,
        "base_model": manager.base_model,
        "device": manager.device,
        "neuronpedia_layers": get_neuronpedia_layers(manager.base_model),
        "sequential_mode": use_sequential_mode(),
    })


@app.route("/api/memory", methods=["GET"])
def get_memory_status():
    """Get current memory status for debugging."""
    try:
        manager = get_manager()
        status = manager.get_memory_status()
        status["sequential_mode"] = use_sequential_mode()
        return jsonify(status)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/unload-llm", methods=["POST"])
def unload_llm():
    """Unload LLM to free GPU memory (for sequential mode)."""
    try:
        manager = get_manager()
        manager.unload_llm()
        return jsonify({"success": True, "message": "LLM unloaded"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/unload-all-saes", methods=["POST"])
def unload_all_saes():
    """Unload all SAEs to free GPU memory."""
    try:
        manager = get_manager()
        manager.unload_all_saes()
        return jsonify({"success": True, "message": "All SAEs unloaded"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/analyze", methods=["POST"])
def analyze_prompt():
    """
    Analyze a prompt and return SAE activations.

    Automatically uses sequential mode for 12b models.

    Request JSON:
        {
            "prompt": "Hello world",
            "top_k": 10,
            "lazy": false,
            "layers": [9, 17]  // optional, defaults to all
        }
    """
    data = request.get_json()
    prompt = data.get("prompt", "")
    top_k = data.get("top_k", 10)
    lazy = data.get("lazy", False)
    layers = data.get("layers", None)

    if not prompt.strip():
        return jsonify({"error": "Prompt cannot be empty"}), 400

    try:
        manager = get_manager()

        # Use sequential mode for 12b models
        if use_sequential_mode():
            if lazy:
                # Lazy mode: just cache residuals and return metadata
                result = manager.gather_and_cache_residuals(prompt, unload_llm_after=True)
            else:
                # Full sequential analysis
                result = manager.analyze_prompt_sequential(prompt, layers=layers, top_k=top_k)
        else:
            # Standard mode for 4b models
            if lazy:
                result = manager.analyze_prompt_lazy(prompt)
            else:
                result = manager.analyze_prompt(prompt, top_k=top_k)

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/analyze/layer", methods=["POST"])
def analyze_layer():
    """
    Analyze a specific layer for a prompt (lazy loading).

    For sequential mode, uses encode_cached_residuals.

    Request JSON:
        {
            "prompt": "Hello world",
            "layer": 9,
            "top_k": 10,
            "cache_key": "optional-for-sequential-mode"
        }
    """
    data = request.get_json()
    prompt = data.get("prompt", "")
    layer = data.get("layer")
    top_k = data.get("top_k", 10)
    cache_key = data.get("cache_key")

    if layer is None:
        return jsonify({"error": "layer is required"}), 400

    try:
        manager = get_manager()

        if use_sequential_mode():
            # Sequential mode: use cache_key if provided, otherwise cache first
            if cache_key is None:
                # Need to cache residuals first
                cache_result = manager.gather_and_cache_residuals(prompt, unload_llm_after=True)
                cache_key = cache_result["cache_key"]

            result = manager.encode_cached_residuals(
                cache_key,
                layer,
                top_k=top_k,
                unload_sae_after=True
            )
        else:
            # Standard lazy mode
            result = manager.analyze_layer(prompt, layer, top_k=top_k)

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/compare", methods=["POST"])
def compare_prompts():
    """
    Compare activations between two prompts.

    Automatically uses sequential mode for 12b models.

    Request JSON:
        {
            "prompt_a": "harmful prompt",
            "prompt_b": "benign prompt",
            "top_k": 50,
            "threshold": 0.1,
            "lazy": false,
            "layers": [9, 17]  // optional
        }
    """
    data = request.get_json()
    prompt_a = data.get("prompt_a", "")
    prompt_b = data.get("prompt_b", "")
    top_k = data.get("top_k", 50)
    threshold = data.get("threshold", 0.1)
    lazy = data.get("lazy", False)
    layers = data.get("layers", None)

    if not prompt_a.strip() or not prompt_b.strip():
        return jsonify({"error": "Both prompts are required"}), 400

    try:
        manager = get_manager()

        if use_sequential_mode():
            if lazy:
                # Cache residuals for both prompts
                result_a = manager.gather_and_cache_residuals(prompt_a, unload_llm_after=False)
                result_b = manager.gather_and_cache_residuals(prompt_b, unload_llm_after=True)
                result = {
                    "prompt_a": prompt_a,
                    "prompt_b": prompt_b,
                    "tokens_a": result_a["tokens"],
                    "tokens_b": result_b["tokens"],
                    "cache_key_a": result_a["cache_key"],
                    "cache_key_b": result_b["cache_key"],
                    "available_layers": manager.sae_layers,
                }
            else:
                # Full sequential comparison
                result = manager.compare_prompts_sequential(
                    prompt_a, prompt_b,
                    layers=layers,
                    top_k=top_k,
                    threshold=threshold
                )
        else:
            # Standard mode
            if lazy:
                result = manager.compare_prompts_lazy(prompt_a, prompt_b)
            else:
                result = manager.compare_prompts(prompt_a, prompt_b, top_k=top_k, threshold=threshold)

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/compare/layer", methods=["POST"])
def compare_layer():
    """
    Compare activations for a specific layer.

    Request JSON:
        {
            "prompt_a": "harmful prompt",
            "prompt_b": "benign prompt",
            "layer": 9,
            "top_k": 50,
            "threshold": 0.1,
            "cache_key_a": "optional",
            "cache_key_b": "optional"
        }
    """
    data = request.get_json()
    prompt_a = data.get("prompt_a", "")
    prompt_b = data.get("prompt_b", "")
    layer = data.get("layer")
    top_k = data.get("top_k", 50)
    threshold = data.get("threshold", 0.1)
    cache_key_a = data.get("cache_key_a")
    cache_key_b = data.get("cache_key_b")

    if layer is None:
        return jsonify({"error": "layer is required"}), 400

    try:
        manager = get_manager()

        if use_sequential_mode():
            # Ensure residuals are cached
            if cache_key_a is None:
                result_a = manager.gather_and_cache_residuals(prompt_a, unload_llm_after=False)
                cache_key_a = result_a["cache_key"]
            if cache_key_b is None:
                result_b = manager.gather_and_cache_residuals(prompt_b, unload_llm_after=True)
                cache_key_b = result_b["cache_key"]

            # Get cached data
            from sae import _residual_cache
            cached_a = _residual_cache.get(cache_key_a, {})
            cached_b = _residual_cache.get(cache_key_b, {})

            tokens_a = cached_a.get("tokens", [])
            tokens_b = cached_b.get("tokens", [])

            # Load residuals and encode
            import torch
            residuals_a = cached_a["residuals"][layer].to(manager.device)
            residuals_b = cached_b["residuals"][layer].to(manager.device)

            # Load SAE
            sae = manager.get_sae(layer)

            # Encode
            sae_acts_a = sae.encode(residuals_a.to(torch.float32))
            sae_acts_b = sae.encode(residuals_b.to(torch.float32))

            # Compute differential
            mean_acts_a = sae_acts_a[1:].mean(dim=0)
            mean_acts_b = sae_acts_b[1:].mean(dim=0)
            diff = mean_acts_a - mean_acts_b
            epsilon = 1e-6
            ratio = mean_acts_a / (mean_acts_b + epsilon)

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
                        "neuronpedia_url": manager.get_neuronpedia_url(feat_idx, layer),
                    })

                top_feature_ids = [f["feature_id"] for f in differential_features]
                for feat_id in top_feature_ids:
                    token_acts_a[feat_id] = [round(v, 4) for v in sae_acts_a[:, feat_id].tolist()]
                    token_acts_b[feat_id] = [round(v, 4) for v in sae_acts_b[:, feat_id].tolist()]

            # Unload SAE
            manager.unload_sae(layer)

            result = {
                "layer": layer,
                "tokens_a": tokens_a,
                "tokens_b": tokens_b,
                "differential_features": differential_features,
                "token_activations_a": token_acts_a,
                "token_activations_b": token_acts_b,
            }
        else:
            # Standard mode
            result = manager.compare_prompts_layer(
                prompt_a, prompt_b, layer,
                top_k=top_k,
                threshold=threshold
            )

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/config", methods=["POST"])
def update_config():
    """Update SAE configuration and reload models."""
    data = request.get_json()

    model_path = data.get("model_path")
    sae_repo = data.get("sae_repo")
    sae_width = data.get("sae_width")
    sae_l0 = data.get("sae_l0")
    base_model = data.get("base_model")

    try:
        manager = get_manager()
        new_config = manager.reconfigure(
            model_path=model_path,
            sae_repo=sae_repo,
            sae_width=sae_width,
            sae_l0=sae_l0,
            base_model=base_model,
        )
        return jsonify({"success": True, "config": new_config})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/apply-steering-permanent", methods=["POST"])
def apply_steering_permanent():
    """
    Apply steering vectors permanently to model weights and save.

    Request JSON:
        {
            "features": [
                {"layer": 17, "feature_id": 12345, "coefficient": 0.5},
                {"layer": 17, "feature_id": 67890, "coefficient": -0.3}
            ],
            "output_path": "/path/to/save/modified-model",
            "scale_factor": 1.0
        }
    """
    data = request.get_json()

    features = data.get("features", [])
    output_path = data.get("output_path")
    scale_factor = data.get("scale_factor", 1.0)

    if not features:
        return jsonify({"success": False, "error": "No features provided"}), 400

    if not output_path:
        return jsonify({"success": False, "error": "No output_path provided"}), 400

    try:
        manager = get_manager()
        result = manager.apply_steering_to_weights(
            features=features,
            output_path=output_path,
            scale_factor=scale_factor,
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/unload-sae/<int:layer>", methods=["POST"])
def unload_sae(layer):
    """Unload SAE weights for a specific layer to free memory."""
    try:
        manager = get_manager()
        manager.unload_sae(layer)
        return jsonify({"success": True, "message": f"Layer {layer} SAE unloaded"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/rank-features", methods=["POST"])
def rank_features():
    """
    Rank features across multiple prompt pairs.

    Automatically uses sequential mode for 12b models (caches residuals to CPU,
    unloads LLM, then processes with SAE).

    Request JSON:
        {
            "prompt_pairs": [
                {"harmful": "...", "benign": "..."},
                ...
            ],
            "top_k": 100,
            "lazy": true
        }
    """
    data = request.get_json()
    prompt_pairs = data.get("prompt_pairs", [])
    top_k = data.get("top_k", 100)
    lazy = data.get("lazy", False)

    if not prompt_pairs:
        return jsonify({"error": "At least one prompt pair is required"}), 400

    for i, pair in enumerate(prompt_pairs):
        if "harmful" not in pair or "benign" not in pair:
            return jsonify({"error": f"Pair {i} missing 'harmful' or 'benign' field"}), 400

    try:
        manager = get_manager()

        if use_sequential_mode():
            # Sequential mode: batch cache all residuals first, then unload LLM
            all_prompts = []
            for pair in prompt_pairs:
                all_prompts.append(pair["harmful"])
                all_prompts.append(pair["benign"])

            # Cache all residuals to CPU, then unload LLM
            batch_result = manager.gather_and_cache_residuals_batch(all_prompts, unload_llm_after=True)

            # Build cache key mapping for the lazy result
            # Convert string keys to integers for rank_features_layer compatibility
            cache_keys = [int(k) for k in batch_result["cache_keys"]]
            cache_entries = []
            for i in range(0, len(cache_keys), 2):
                cache_entries.append({
                    "harmful_key": cache_keys[i],
                    "benign_key": cache_keys[i + 1]
                })

            # Store in residual cache for rank_features_layer to use
            # Use integer hash key to match existing rank_features_layer implementation
            from sae import _residual_cache
            master_key = hash(f"rank_seq_{len(prompt_pairs)}_{hash(str(prompt_pairs[:2]))}")
            _residual_cache[master_key] = {
                "mode": "pairs",
                "entries": cache_entries,
                "num_pairs": len(prompt_pairs),
            }

            result = {
                "cache_key": str(master_key),  # String to avoid JS integer precision loss
                "num_prompt_pairs": len(prompt_pairs),
                "available_layers": manager.sae_layers,
                "sequential_mode": True,
            }
        else:
            # Standard mode
            if lazy:
                result = manager.rank_features_lazy(prompt_pairs=prompt_pairs)
            else:
                result = manager.rank_features_for_refusal(prompt_pairs, top_k=top_k)

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/rank-features/layer", methods=["POST"])
def rank_features_layer():
    """
    Rank features for a SINGLE layer using cached residuals.

    Request JSON:
        {
            "cache_key": 12345,
            "layer": 9,
            "top_k": 100
        }
    """
    data = request.get_json()
    cache_key = data.get("cache_key")
    layer = data.get("layer")
    top_k = data.get("top_k", 100)

    if cache_key is None:
        return jsonify({"error": "cache_key is required"}), 400

    if layer is None:
        return jsonify({"error": "layer is required"}), 400

    try:
        manager = get_manager()
        result = manager.rank_features_layer(cache_key, layer, top_k=top_k)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/rank-features/token-activations", methods=["POST"])
def get_feature_token_activations():
    """
    Get token-level activations for a specific feature across all cached prompts.

    Request JSON:
        {
            "cache_key": "12345",
            "layer": 9,
            "feature_id": 12345
        }
    """
    data = request.get_json()
    cache_key = data.get("cache_key")
    layer = data.get("layer")
    feature_id = data.get("feature_id")

    if cache_key is None:
        return jsonify({"error": "cache_key is required"}), 400

    if layer is None:
        return jsonify({"error": "layer is required"}), 400

    if feature_id is None:
        return jsonify({"error": "feature_id is required"}), 400

    try:
        manager = get_manager()
        result = manager.get_feature_token_activations_for_ranking(cache_key, layer, feature_id)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/rank-features-single", methods=["POST"])
def rank_features_single():
    """
    Rank features by activation strength on a single category of prompts.

    Automatically uses sequential mode for 12b models.

    Request JSON:
        {
            "prompts": ["prompt1", "prompt2", ...],
            "category": "harmful" or "harmless",
            "top_k": 100,
            "lazy": true
        }
    """
    data = request.get_json()
    prompts = data.get("prompts", [])
    category = data.get("category", "harmful")
    top_k = data.get("top_k", 100)
    lazy = data.get("lazy", False)

    if not prompts:
        return jsonify({"error": "At least one prompt is required"}), 400

    if category not in ("harmful", "harmless"):
        return jsonify({"error": "Category must be 'harmful' or 'harmless'"}), 400

    try:
        manager = get_manager()

        if use_sequential_mode():
            # Sequential mode: batch cache all residuals first, then unload LLM
            batch_result = manager.gather_and_cache_residuals_batch(prompts, unload_llm_after=True)

            # Store in residual cache for rank_features_layer to use
            # Use integer hash key to match existing rank_features_layer implementation
            # Convert string keys to integers for rank_features_layer compatibility
            from sae import _residual_cache
            master_key = hash(f"rank_single_seq_{category}_{len(prompts)}_{hash(str(prompts[:2]))}")
            cache_keys_int = [int(k) for k in batch_result["cache_keys"]]
            _residual_cache[master_key] = {
                "mode": "single",
                "category": category,
                "entries": cache_keys_int,
                "num_prompts": len(prompts),
            }

            result = {
                "cache_key": str(master_key),  # String to avoid JS integer precision loss
                "num_prompts": len(prompts),
                "category": category,
                "available_layers": manager.sae_layers,
                "sequential_mode": True,
            }
        else:
            # Standard mode
            if lazy:
                result = manager.rank_features_lazy(prompts=prompts, category=category)
            else:
                result = manager.rank_features_single_category(prompts, category=category, top_k=top_k)

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/export/rankings", methods=["POST"])
def export_rankings():
    """
    Export feature rankings as JSON or CSV.

    Request JSON:
        {
            "data": { ... ranking result ... },
            "format": "json" or "csv"
        }
    """
    data = request.get_json()
    ranking_data = data.get("data", {})
    export_format = data.get("format", "json")

    if export_format == "json":
        response = Response(
            json.dumps(ranking_data, indent=2),
            mimetype="application/json",
            headers={"Content-Disposition": f"attachment;filename=rankings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"}
        )
        return response

    elif export_format == "csv":
        output = io.StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow(["Layer", "Rank", "FeatureID", "ConsistencyScore", "MeanHarmful", "MeanBenign", "DifferentialScore", "NeuronpediaURL"])

        # Data rows
        for layer, layer_data in ranking_data.get("layers", {}).items():
            for rank, feat in enumerate(layer_data.get("ranked_features", []), 1):
                writer.writerow([
                    layer,
                    rank,
                    feat.get("feature_id"),
                    feat.get("consistency_score"),
                    feat.get("mean_harmful_activation"),
                    feat.get("mean_benign_activation"),
                    feat.get("differential_score"),
                    feat.get("neuronpedia_url"),
                ])

        response = Response(
            output.getvalue(),
            mimetype="text/csv",
            headers={"Content-Disposition": f"attachment;filename=rankings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"}
        )
        return response

    return jsonify({"error": "Invalid format. Use 'json' or 'csv'"}), 400


if __name__ == "__main__":
    print("=" * 60)
    print("LM Feature Studio - Batch Ranking")
    print("=" * 60)

    if PRELOAD_MODELS:
        print("Pre-loading models (PRELOAD_MODELS=1)...")
        initialize_models()
    else:
        print("Models will load on first request.")
        print("Set PRELOAD_MODELS=1 to pre-load at startup.")

    print()
    print("Starting Flask server...")
    print("Open http://127.0.0.1:5000 in your browser")
    print("=" * 60)

    app.run(debug=True, host="127.0.0.1", port=5000)
