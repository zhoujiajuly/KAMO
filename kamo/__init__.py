"""KAMO: bounded-influence probabilistic multi-horizon forecasting."""

from .model import DLinear, GRUForecaster, KAMO, KAMOV5

__version__ = "0.1.0"

__all__ = ["KAMO", "KAMOV5", "DLinear", "GRUForecaster", "__version__"]
