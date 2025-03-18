import joblib
from pathlib import Path

# Constants
MODEL_DIR = Path("models")
FEATURE_NAMES = ['cost', 'productivity', 'area', 'value']

def load_models():
    """
    Load and return the scaler, MLP model, and K-means model
    
    Returns:
        tuple: (scaler, mlp, kmeans) - The loaded models
    """
    try:
        scaler = joblib.load(MODEL_DIR / 'standard_scaler.joblib')
        mlp = joblib.load(MODEL_DIR / 'mlp.joblib')
        kmeans = joblib.load(MODEL_DIR / 'kmeans' / 'k4.joblib')
        return scaler, mlp, kmeans
    except FileNotFoundError as e:
        raise FileNotFoundError(
            "Model files not found. Please ensure all required models "
            "(standard_scaler.joblib, mlp.joblib, and kmeans/k4.joblib) "
            "are in the correct directory."
        ) from e 