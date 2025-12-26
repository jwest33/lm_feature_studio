"""
SAE Configuration

Configuration constants and layer mappings for SAE model management.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Neuronpedia configuration
NEURONPEDIA_API_KEY = os.getenv("NEURONPEDIA_API_KEY")
NEURONPEDIA_BASE_URL = "https://www.neuronpedia.org/api"

# Model path - set via environment variable or defaults to HuggingFace ID
# For local models, set MODEL_PATH in .env or environment
MODEL_PATH = os.getenv("MODEL_PATH", "google/gemma-3-4b-it")

# SAE Configuration
SAE_REPO = "google/gemma-scope-2-4b-it"
SAE_WIDTH = "262k"
SAE_L0 = "small"
BASE_MODEL = "4b"  # Explicit base model selection: "4b" or "12b"

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


def get_available_layers(base_model: str) -> list[int]:
    """Get available SAE layers based on explicit base model selection."""
    if base_model in SAE_LAYERS_BY_MODEL:
        return SAE_LAYERS_BY_MODEL[base_model]
    # Default to 4b layers
    return SAE_LAYERS_BY_MODEL["4b"]


def get_neuronpedia_layers(base_model: str) -> list[int]:
    """Get layers that have Neuronpedia data available based on base model."""
    if base_model in NEURONPEDIA_LAYERS_BY_MODEL:
        return NEURONPEDIA_LAYERS_BY_MODEL[base_model]
    return NEURONPEDIA_LAYERS_BY_MODEL["4b"]


# Computed defaults
SAE_LAYERS = get_available_layers(BASE_MODEL)
NEURONPEDIA_LAYERS = get_neuronpedia_layers(BASE_MODEL)
