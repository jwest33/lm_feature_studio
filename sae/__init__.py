"""
LM Feature Studio

A modular package for exploring Sparse Autoencoder features on Gemma models.
"""

from .manager import SAEModelManager, get_manager, initialize_models
from .sae import JumpReLUSAE
from .config import (
    MODEL_PATH,
    SAE_REPO,
    SAE_WIDTH,
    SAE_L0,
    BASE_MODEL,
    SAE_LAYERS,
    NEURONPEDIA_LAYERS,
    SAE_LAYERS_BY_MODEL,
    NEURONPEDIA_LAYERS_BY_MODEL,
    NEURONPEDIA_API_KEY,
    NEURONPEDIA_BASE_URL,
    get_available_layers,
    get_neuronpedia_layers,
)
from .cache import (
    _residual_cache,
    _neuronpedia_cache,
    clear_residual_cache,
    clear_residual_cache_entry,
    clear_neuronpedia_cache,
)
from .refusal import REFUSAL_PHRASES

__all__ = [
    # Manager
    "SAEModelManager",
    "get_manager",
    "initialize_models",
    # SAE
    "JumpReLUSAE",
    # Config
    "MODEL_PATH",
    "SAE_REPO",
    "SAE_WIDTH",
    "SAE_L0",
    "BASE_MODEL",
    "SAE_LAYERS",
    "NEURONPEDIA_LAYERS",
    "SAE_LAYERS_BY_MODEL",
    "NEURONPEDIA_LAYERS_BY_MODEL",
    "NEURONPEDIA_API_KEY",
    "NEURONPEDIA_BASE_URL",
    "get_available_layers",
    "get_neuronpedia_layers",
    # Cache
    "_residual_cache",
    "_neuronpedia_cache",
    "clear_residual_cache",
    "clear_residual_cache_entry",
    "clear_neuronpedia_cache",
    # Refusal
    "REFUSAL_PHRASES",
]
