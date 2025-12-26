"""
Neuronpedia Integration Mixin

Methods for interacting with the Neuronpedia API and generating URLs.
"""

import requests
from .config import (
    NEURONPEDIA_API_KEY,
    NEURONPEDIA_BASE_URL,
    get_neuronpedia_layers,
)
from .cache import _neuronpedia_cache


class NeuronpediaMixin:
    """Mixin providing Neuronpedia integration for SAEModelManager."""

    def get_neuronpedia_model_id(self) -> str:
        """Get the Neuronpedia model ID based on base model."""
        # Map base model to Neuronpedia model ID
        if self.base_model == "12b":
            return "gemma-3-12b-it"
        elif self.base_model == "4b":
            return "gemma-3-4b-it"
        return "gemma-3-4b-it"  # Default

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
        return layer in get_neuronpedia_layers(self.base_model)

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
                "supported_layers": get_neuronpedia_layers(self.base_model),
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
