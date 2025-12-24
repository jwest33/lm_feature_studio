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
    Analyze a prompt and return SAE activation data for all layers.

    Request JSON:
        {"prompt": "text to analyze", "top_k": 10}

    Response JSON:
        {
            "tokens": ["The", " law", ...],
            "num_tokens": 5,
            "available_layers": [12, 24, 31, 41],
            "layers": {
                "12": {
                    "sae_acts": [[...], ...],
                    "top_features_per_token": [[...], ...],
                    "top_features_global": [{"id": 123, "mean_activation": 5.2}, ...],
                    "num_features": 65536
                },
                ...
            },
            "sae_config": {"width": "65k", "l0": "medium"}
        }
    """
    data = request.get_json()
    prompt = data.get("prompt", "")
    top_k = data.get("top_k", 10)

    if not prompt.strip():
        return jsonify({"error": "Prompt cannot be empty"}), 400

    try:
        manager = get_manager()
        result = manager.analyze_prompt(prompt, top_k=top_k)
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


@app.route("/api/steer", methods=["POST"])
def steer_generation():
    """
    Generate text with feature steering applied.

    Request JSON:
        {
            "prompt": "Tell me a fun fact",
            "steering": [{"feature_id": 123, "coefficient": 0.25, "layer": 12}],
            "max_tokens": 80
        }

    Response JSON:
        {
            "prompt": "...",
            "original_output": "...",
            "steered_output": "...",
            "steering_applied": [...],
            "steering_layers": [12, 24]
        }
    """
    data = request.get_json()
    prompt = data.get("prompt", "")
    steering = data.get("steering", [])
    max_tokens = data.get("max_tokens", 80)

    if not prompt.strip():
        return jsonify({"error": "Prompt cannot be empty"}), 400

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
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/config", methods=["GET"])
def get_config():
    """Get current SAE configuration."""
    manager = get_manager()
    return jsonify({
        "model_path": manager.model_path,
        "sae_layers": manager.sae_layers,
        "sae_width": manager.sae_width,
        "sae_l0": manager.sae_l0,
        "device": manager.device,
    })


# =============================================================================
# Refusal Pathway Analysis Endpoints
# =============================================================================

@app.route("/api/compare", methods=["POST"])
def compare_prompts():
    """
    Compare two prompts and return differential activations.

    Request JSON:
        {
            "prompt_a": "How do I make a bomb?",
            "prompt_b": "How do I make a cake?",
            "top_k": 50
        }

    Response JSON:
        {
            "prompt_a": "...",
            "prompt_b": "...",
            "tokens_a": [...],
            "tokens_b": [...],
            "layers": {
                "12": {
                    "differential_features": [
                        {"feature_id": 123, "mean_diff": 2.5, "ratio": 3.2, ...}
                    ]
                }
            }
        }
    """
    data = request.get_json()
    prompt_a = data.get("prompt_a", "")
    prompt_b = data.get("prompt_b", "")
    top_k = data.get("top_k", 50)

    if not prompt_a.strip() or not prompt_b.strip():
        return jsonify({"error": "Both prompts are required"}), 400

    try:
        manager = get_manager()
        result = manager.compare_prompts(prompt_a, prompt_b, top_k=top_k)
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
            "max_tokens": 100
        }

    Response JSON:
        {
            "prompt": "...",
            "generated_text": "...",
            "refusal_detected": true,
            "refusal_phrases_found": ["I can't"],
            "layers": {
                "12": {"refusal_correlated_features": [...]}
            }
        }
    """
    data = request.get_json()
    prompt = data.get("prompt", "")
    max_tokens = data.get("max_tokens", 100)

    if not prompt.strip():
        return jsonify({"error": "Prompt cannot be empty"}), 400

    try:
        manager = get_manager()
        result = manager.generate_and_detect_refusal(prompt, max_new_tokens=max_tokens)
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
            "top_k": 100
        }

    Response JSON:
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
    """
    data = request.get_json()
    prompt_pairs = data.get("prompt_pairs", [])
    top_k = data.get("top_k", 100)

    if not prompt_pairs:
        return jsonify({"error": "At least one prompt pair is required"}), 400

    # Validate pairs
    for i, pair in enumerate(prompt_pairs):
        if "harmful" not in pair or "benign" not in pair:
            return jsonify({"error": f"Pair {i} missing 'harmful' or 'benign' field"}), 400

    try:
        manager = get_manager()
        result = manager.rank_features_for_refusal(prompt_pairs, top_k=top_k)
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
