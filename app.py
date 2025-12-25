"""
Flask SAE Feature Explorer

A local web app for exploring SAE features on Gemma models,
replicating the Neuronpedia visualization interface.
"""

import os
import json
import csv
import io
from datetime import datetime
from flask import Flask, render_template, request, jsonify, Response
from sae_model import get_manager, initialize_models

app = Flask(__name__)

# Configuration
PRELOAD_MODELS = os.environ.get("PRELOAD_MODELS", "0") == "1"


# =============================================================================
# Routes
# =============================================================================

@app.route("/")
def index():
    """Serve the main UI."""
    return render_template("index.html")


@app.route("/api/analyze", methods=["POST"])
def analyze_prompt():
    """
    Analyze a prompt - lazy mode returns only tokens, eager mode returns all layers.

    Request JSON:
        {"prompt": "text to analyze", "top_k": 10, "lazy": true}

    With lazy=true (default):
        Returns tokens and available layers, NO SAE data.
        Use /api/analyze/layer to fetch individual layer data.

    With lazy=false:
        Returns full analysis with all layers (loads all SAE weights).

    Response JSON (lazy=true):
        {
            "tokens": ["The", " law", ...],
            "num_tokens": 5,
            "available_layers": [9, 17, 22, 29],
            "sae_config": {"width": "65k", "l0": "medium"}
        }
    """
    data = request.get_json()
    prompt = data.get("prompt", "")
    top_k = data.get("top_k", 10)
    lazy = data.get("lazy", True)  # Default to lazy loading

    if not prompt.strip():
        return jsonify({"error": "Prompt cannot be empty"}), 400

    try:
        manager = get_manager()
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
    Analyze a specific layer for a prompt. Loads SAE weights on-demand.

    Request JSON:
        {"prompt": "text to analyze", "layer": 9, "top_k": 10}

    Response JSON:
        {
            "layer": 9,
            "sae_acts": [[...], ...],
            "top_features_per_token": [[...], ...],
            "top_features_global": [{"id": 123, "mean_activation": 5.2}, ...],
            "num_features": 262144
        }
    """
    data = request.get_json()
    prompt = data.get("prompt", "")
    layer = data.get("layer")
    top_k = data.get("top_k", 10)

    if not prompt.strip():
        return jsonify({"error": "Prompt cannot be empty"}), 400

    if layer is None:
        return jsonify({"error": "Layer is required"}), 400

    try:
        manager = get_manager()
        result = manager.analyze_layer(prompt, layer, top_k=top_k)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/feature/<int:feature_id>", methods=["POST"])
def get_feature_info(feature_id: int):
    """
    Get activation pattern for a specific feature.

    Request JSON:
        {"prompt": "text to analyze", "layer": 12}

    Response JSON:
        {
            "feature_id": 123,
            "layer": 12,
            "tokens": [...],
            "activations": [...],
            "normalized_activations": [...],
            "top_tokens": [{"token": "energy", "logit": 5.2}, ...]
        }
    """
    data = request.get_json()
    prompt = data.get("prompt", "")
    layer = data.get("layer")  # Optional, defaults to first available layer

    if not prompt.strip():
        return jsonify({"error": "Prompt cannot be empty"}), 400

    try:
        manager = get_manager()

        # Get activations for this feature at specified layer
        result = manager.get_feature_activations(prompt, feature_id, layer=layer)

        # Get top predicted tokens for this feature
        result["top_tokens"] = manager.get_top_logits_for_feature(feature_id, layer=layer, top_k=10)

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/neuronpedia/<int:layer>/<int:feature_id>", methods=["GET"])
def get_neuronpedia_data(layer: int, feature_id: int):
    """
    Fetch feature data from Neuronpedia.

    Returns explanations, lists, positive/negative logits, and top activations
    for the specified feature.

    Response JSON:
        {
            "feature_id": 123,
            "layer": 12,
            "neuronpedia_url": "https://...",
            "explanations": [{"description": "...", "score": 0.8}],
            "positive_logits": [{"token": "energy", "value": 1.23}],
            "negative_logits": [{"token": "the", "value": -0.5}],
            "top_activations": [{"tokens": [...], "values": [...], "max_value": 5.2}],
            "max_activation": 15.5,
            "frac_nonzero": 0.003
        }
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
            "normalization": "preserve_norm",  // Optional: "preserve_norm", "clamp", or null
            "norm_clamp_factor": 1.5,          // Optional: for "clamp" mode
            "unit_normalize": true,            // Optional: normalize decoder vectors to unit norm
            "skip_baseline": false             // Optional: skip baseline generation
        }

    Response JSON:
        {
            "prompt": "...",
            "original_output": "..." or null,  // null if skip_baseline is true
            "steered_output": "...",
            "steering_applied": [...],
            "steering_layers": [12, 24],
            "normalization": "preserve_norm",
            "unit_normalize": true
        }
    """
    data = request.get_json()
    prompt = data.get("prompt", "")
    steering = data.get("steering", [])
    max_tokens = data.get("max_tokens", 80)
    normalization = data.get("normalization", None)
    norm_clamp_factor = data.get("norm_clamp_factor", 1.5)
    unit_normalize = data.get("unit_normalize", False)
    skip_baseline = data.get("skip_baseline", False)

    if not prompt.strip():
        return jsonify({"error": "Prompt cannot be empty"}), 400

    # Validate normalization mode
    valid_modes = [None, "preserve_norm", "clamp"]
    if normalization not in valid_modes:
        return jsonify({"error": f"Invalid normalization mode. Use one of: {valid_modes}"}), 400

    # Validate steering config
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
    from sae_model import get_neuronpedia_layers
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
    })


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
            "scale_factor": 1.0  # Optional, default 1.0
        }

    Response:
        {
            "success": true,
            "output_path": "/path/to/saved/model",
            "modifications": [...],
            "total_features": 2
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


# =============================================================================
# Refusal Pathway Analysis Endpoints
# =============================================================================

@app.route("/api/unload-sae/<int:layer>", methods=["POST"])
def unload_sae(layer):
    """
    Unload SAE weights for a specific layer to free memory.

    Response JSON:
        {"success": true, "message": "Layer 9 SAE unloaded"}
    """
    try:
        manager = get_manager()
        manager.unload_sae(layer)
        return jsonify({"success": True, "message": f"Layer {layer} SAE unloaded"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/compare", methods=["POST"])
def compare_prompts():
    """
    Compare two prompts and return differential activations.

    Request JSON:
        {
            "prompt_a": "How do I make a bomb?",
            "prompt_b": "How do I make a cake?",
            "top_k": 50,
            "lazy": true  // Optional: if true, only tokenize and return available layers
        }

    Response JSON (lazy=false, default):
        {
            "prompt_a": "...",
            "prompt_b": "...",
            "tokens_a": [...],
            "tokens_b": [...],
            "layers": {...}
        }

    Response JSON (lazy=true):
        {
            "prompt_a": "...",
            "prompt_b": "...",
            "tokens_a": [...],
            "tokens_b": [...],
            "available_layers": [9, 17, 22, 29]
        }
    """
    data = request.get_json()
    prompt_a = data.get("prompt_a", "")
    prompt_b = data.get("prompt_b", "")
    top_k = data.get("top_k", 50)
    lazy = data.get("lazy", False)

    if not prompt_a.strip() or not prompt_b.strip():
        return jsonify({"error": "Both prompts are required"}), 400

    try:
        manager = get_manager()
        if lazy:
            result = manager.compare_prompts_lazy(prompt_a, prompt_b)
        else:
            result = manager.compare_prompts(prompt_a, prompt_b, top_k=top_k)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/compare/layer", methods=["POST"])
def compare_prompts_layer():
    """
    Compare two prompts for a SINGLE layer. Loads SAE weights on-demand.

    Request JSON:
        {
            "prompt_a": "How do I make a bomb?",
            "prompt_b": "How do I make a cake?",
            "layer": 9,
            "top_k": 50
        }

    Response JSON:
        {
            "layer": 9,
            "tokens_a": [...],
            "tokens_b": [...],
            "differential_features": [...],
            "token_activations_a": {...},
            "token_activations_b": {...}
        }
    """
    data = request.get_json()
    prompt_a = data.get("prompt_a", "")
    prompt_b = data.get("prompt_b", "")
    layer = data.get("layer")
    top_k = data.get("top_k", 50)

    if not prompt_a.strip() or not prompt_b.strip():
        return jsonify({"error": "Both prompts are required"}), 400

    if layer is None:
        return jsonify({"error": "Layer is required"}), 400

    try:
        manager = get_manager()
        result = manager.compare_prompts_layer(prompt_a, prompt_b, layer, top_k=top_k)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/detect-refusal", methods=["POST"])
def detect_refusal():
    """
    Generate response and analyze for refusal patterns.

    Request JSON:
        {
            "prompt": "How do I hack into...",
            "max_tokens": 100,
            "lazy": true  // Optional: if true, generate text but don't analyze layers
        }

    Response JSON (lazy=false, default):
        {
            "prompt": "...",
            "generated_text": "...",
            "refusal_detected": true,
            "refusal_phrases_found": ["I can't"],
            "layers": {
                "12": {"refusal_correlated_features": [...]}
            }
        }

    Response JSON (lazy=true):
        {
            "prompt": "...",
            "generated_text": "...",
            "refusal_detected": true,
            "refusal_phrases_found": ["I can't"],
            "cache_key": 12345,
            "available_layers": [9, 17, 22, 29]
        }
    """
    data = request.get_json()
    prompt = data.get("prompt", "")
    max_tokens = data.get("max_tokens", 100)
    lazy = data.get("lazy", False)

    if not prompt.strip():
        return jsonify({"error": "Prompt cannot be empty"}), 400

    try:
        manager = get_manager()
        if lazy:
            result = manager.detect_refusal_lazy(prompt, max_new_tokens=max_tokens)
        else:
            result = manager.generate_and_detect_refusal(prompt, max_new_tokens=max_tokens)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/detect-refusal/layer", methods=["POST"])
def detect_refusal_layer():
    """
    Analyze refusal-correlated features for a SINGLE layer.

    Request JSON:
        {
            "cache_key": 12345,
            "layer": 9
        }

    Response JSON:
        {
            "layer": 9,
            "refusal_correlated_features": [...]
        }
    """
    data = request.get_json()
    cache_key = data.get("cache_key")
    layer = data.get("layer")

    if cache_key is None:
        return jsonify({"error": "cache_key is required"}), 400

    if layer is None:
        return jsonify({"error": "layer is required"}), 400

    try:
        manager = get_manager()
        result = manager.detect_refusal_layer(cache_key, layer)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/rank-features", methods=["POST"])
def rank_features():
    """
    Rank features across multiple prompt pairs.

    Request JSON:
        {
            "prompt_pairs": [
                {"harmful": "...", "benign": "..."},
                ...
            ],
            "top_k": 100,
            "lazy": true  // Optional: if true, cache residuals but don't analyze layers
        }

    Response JSON (lazy=false, default):
        {
            "num_prompt_pairs": 5,
            "layers": {
                "12": {
                    "ranked_features": [
                        {"feature_id": 123, "consistency_score": 0.8, ...}
                    ]
                }
            }
        }

    Response JSON (lazy=true):
        {
            "cache_key": 12345,
            "num_prompt_pairs": 5,
            "available_layers": [9, 17, 22, 29]
        }
    """
    data = request.get_json()
    prompt_pairs = data.get("prompt_pairs", [])
    top_k = data.get("top_k", 100)
    lazy = data.get("lazy", False)

    if not prompt_pairs:
        return jsonify({"error": "At least one prompt pair is required"}), 400

    # Validate pairs
    for i, pair in enumerate(prompt_pairs):
        if "harmful" not in pair or "benign" not in pair:
            return jsonify({"error": f"Pair {i} missing 'harmful' or 'benign' field"}), 400

    try:
        manager = get_manager()
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

    Response JSON:
        {
            "layer": 9,
            "ranked_features": [...]
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


@app.route("/api/rank-features-single", methods=["POST"])
def rank_features_single():
    """
    Rank features by activation strength on a single category of prompts.

    Request JSON:
        {
            "prompts": ["prompt1", "prompt2", ...],
            "category": "harmful" or "harmless",
            "top_k": 100,
            "lazy": true  // Optional: if true, cache residuals but don't analyze layers
        }

    Response JSON (lazy=false, default):
        {
            "num_prompts": 10,
            "category": "harmful",
            "layers": {
                "12": {
                    "ranked_features": [
                        {"feature_id": 123, "mean_activation": 0.5, ...}
                    ]
                }
            }
        }

    Response JSON (lazy=true):
        {
            "cache_key": 12345,
            "num_prompts": 10,
            "category": "harmful",
            "available_layers": [9, 17, 22, 29]
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
        if lazy:
            result = manager.rank_features_lazy(prompts=prompts, category=category)
        else:
            result = manager.rank_features_single_category(prompts, category=category, top_k=top_k)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/export/comparison", methods=["POST"])
def export_comparison():
    """
    Export comparison results as JSON or CSV.

    Request JSON:
        {
            "data": { ... comparison result ... },
            "format": "json" or "csv"
        }
    """
    data = request.get_json()
    comparison_data = data.get("data", {})
    export_format = data.get("format", "json")

    if export_format == "json":
        response = Response(
            json.dumps(comparison_data, indent=2),
            mimetype="application/json",
            headers={"Content-Disposition": f"attachment;filename=comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"}
        )
        return response

    elif export_format == "csv":
        output = io.StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow(["Layer", "FeatureID", "MeanDiff", "ActivationA", "ActivationB", "Ratio", "NeuronpediaURL"])

        # Data rows
        for layer, layer_data in comparison_data.get("layers", {}).items():
            for feat in layer_data.get("differential_features", []):
                writer.writerow([
                    layer,
                    feat.get("feature_id"),
                    feat.get("mean_diff"),
                    feat.get("activation_a"),
                    feat.get("activation_b"),
                    feat.get("ratio"),
                    feat.get("neuronpedia_url"),
                ])

        response = Response(
            output.getvalue(),
            mimetype="text/csv",
            headers={"Content-Disposition": f"attachment;filename=comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"}
        )
        return response

    return jsonify({"error": "Invalid format. Use 'json' or 'csv'"}), 400


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


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SAE Feature Explorer")
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
