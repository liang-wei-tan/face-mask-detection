from models.baseline_cnn import create_baseline_cnn as _create_baseline_cnn
from models.baseline_cnn_64 import create_baseline_cnn as _create_baseline_cnn_64
from models.transfer_inception import create_transfer_inception as _create_transfer_inception
from models.transfer_inception_unfrozen import create_transfer_inception_unfrozen as _create_transfer_inception_unfrozen
from models.transfer_inception_unfrozen import create_transfer_dino_v2 as _create_transfer_dino_v2

# Model registry
MODELS = {
    "baseline_cnn": _create_baseline_cnn,
    "baseline_cnn_64": _create_baseline_cnn_64,
    "transfer_inception": _create_transfer_inception,
    "transfer_inception_unfrozen": _create_transfer_inception_unfrozen,
    "transfer_dino_v2": _create_transfer_dino_v2, 
}

__all__ = ["MODELS", "get_model"]


def get_model(model_name):
    """
    Get a model by name.

    Args:
        model_name: Name of the model (e.g., 'baseline_cnn', 'baseline_cnn_64', 'transfer_inception')

    Returns:
        The model creation function

    Raises:
        ValueError: If model name not found
    """
    if model_name not in MODELS:
        raise ValueError(
            f"Model '{model_name}' not found. Available models: {list(MODELS.keys())}"
        )
    return MODELS[model_name]
