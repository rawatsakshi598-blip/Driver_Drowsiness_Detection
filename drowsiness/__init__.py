"""Driver Drowsiness Detection — production package."""
from .config import Config, load_config, get_config, FaceMeshIndices

__all__ = ["Config", "load_config", "get_config", "FaceMeshIndices"]
__version__ = "1.0.0"
