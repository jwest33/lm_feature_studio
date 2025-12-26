"""
Cache Management

Global caches for residual activations and Neuronpedia data.
"""

import torch

# Cache for Neuronpedia data to avoid repeated API calls
_neuronpedia_cache: dict[str, dict] = {}

# Cache for residual activations to avoid re-running model
_residual_cache: dict[str, dict] = {}


def clear_residual_cache():
    """Clear all residual activation caches."""
    global _residual_cache
    _residual_cache.clear()


def clear_residual_cache_entry(cache_key: int | str):
    """Clear a specific entry from the residual cache."""
    if isinstance(cache_key, str):
        cache_key = int(cache_key)
    if cache_key in _residual_cache:
        del _residual_cache[cache_key]
        print(f"Cleared cache entry: {cache_key}")


def clear_neuronpedia_cache():
    """Clear all Neuronpedia data caches."""
    global _neuronpedia_cache
    _neuronpedia_cache.clear()


def get_memory_status(device: str, saes: dict, model) -> dict:
    """
    Get current memory status for debugging sequential loading.

    Args:
        device: The device being used (cuda/cpu)
        saes: Dictionary of loaded SAEs
        model: The loaded model (or None)

    Returns:
        Dictionary with memory usage information
    """
    status = {
        "llm_loaded": model is not None,
        "saes_loaded": list(saes.keys()),
        "num_cached_residuals": len(_residual_cache),
        "device": device,
    }

    if torch.cuda.is_available():
        status["gpu_allocated_mb"] = round(torch.cuda.memory_allocated() / 1024 / 1024, 2)
        status["gpu_reserved_mb"] = round(torch.cuda.memory_reserved() / 1024 / 1024, 2)
        status["gpu_max_allocated_mb"] = round(torch.cuda.max_memory_allocated() / 1024 / 1024, 2)

    return status
