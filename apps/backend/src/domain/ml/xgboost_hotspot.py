"""
XGBoost Model for Urban Waterlogging Hotspot Prediction (Inference Only).

Lightweight version for backend embedding - loads pre-trained model
and provides prediction capabilities only. Training is done separately.

Features:
- 18-dimensional input (terrain, rainfall, land cover, SAR, temporal)
- Binary classification (flood-prone vs. safe)
- Achieves AUC 0.98 on Delhi hotspots dataset
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

logger = logging.getLogger(__name__)

# Feature names for the 18-dimensional input
FEATURE_NAMES = [
    # Terrain (6)
    "elevation",
    "slope",
    "tpi",       # Topographic Position Index
    "tri",       # Terrain Ruggedness Index
    "twi",       # Topographic Wetness Index
    "spi",       # Stream Power Index
    # Precipitation (5)
    "rainfall_24h",
    "rainfall_3d",
    "rainfall_7d",
    "max_daily_7d",
    "wet_days_7d",
    # Land Cover (2)
    "impervious_pct",
    "built_up_pct",
    # SAR (4)
    "sar_vv_mean",
    "sar_vh_mean",
    "sar_vv_vh_ratio",
    "sar_change_mag",
    # Temporal (1)
    "is_monsoon",
]


class XGBoostHotspotModel:
    """
    XGBoost classifier for waterlogging hotspot prediction (inference only).

    This model predicts the susceptibility of a location to waterlogging
    based on terrain, precipitation, land cover, and temporal features.

    Performance on Delhi hotspots:
    - AUC: 0.98
    - Recall: 0.97
    - Precision: 0.92
    """

    def __init__(self, model_name: str = "xgboost_hotspot"):
        """
        Initialize XGBoost hotspot model.

        Args:
            model_name: Name identifier for the model
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError(
                "XGBoost is required. Install with: pip install xgboost"
            )

        self.model_name = model_name
        self.model: Optional[xgb.XGBClassifier] = None
        self.feature_names = FEATURE_NAMES.copy()
        self.params: Dict = {}
        self.cv_results: Optional[Dict] = None
        self.feature_importance: Optional[Dict[str, float]] = None
        self._trained = False

    @property
    def is_trained(self) -> bool:
        """Check if model is loaded and ready for inference."""
        return self._trained and self.model is not None

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict binary class labels.

        Args:
            X: Feature matrix of shape (n_samples, 18)

        Returns:
            Binary predictions (0 = safe, 1 = flood-prone)
        """
        if not self.is_trained:
            raise RuntimeError("Model must be loaded before prediction")

        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict flood susceptibility probability.

        Args:
            X: Feature matrix of shape (n_samples, 18)

        Returns:
            Probability of positive class (flood-prone), shape (n_samples,)
        """
        if not self.is_trained:
            raise RuntimeError("Model must be loaded before prediction")

        return self.model.predict_proba(X)[:, 1]

    def load(self, path: Path) -> "XGBoostHotspotModel":
        """
        Load pre-trained model from disk.

        Args:
            path: Directory containing xgboost_model.json and metadata.json

        Returns:
            self (for method chaining)
        """
        path = Path(path)

        # Load XGBoost model
        model_file = path / "xgboost_model.json"
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")

        self.model = xgb.XGBClassifier()
        self.model.load_model(str(model_file))

        # Load metadata
        metadata_file = path / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            self.model_name = metadata.get("model_name", self.model_name)
            self.feature_names = metadata.get("feature_names", FEATURE_NAMES)
            self.params = metadata.get("params", {})
            self.cv_results = metadata.get("cv_results")
            self.feature_importance = metadata.get("feature_importance")
            self._trained = metadata.get("trained", True)
        else:
            # Assume model is trained if file exists
            self._trained = True

        logger.info(f"XGBoost model loaded from {path}")
        return self

    def get_model_info(self) -> Dict:
        """Return model metadata for debugging/monitoring."""
        return {
            "model_name": self.model_name,
            "model_type": "XGBoostHotspotModel",
            "is_trained": self.is_trained,
            "n_features": len(self.feature_names),
            "feature_names": self.feature_names,
            "params": self.params,
            "cv_results": self.cv_results,
            "feature_importance": self.feature_importance,
        }


def get_risk_level(probability: float) -> Tuple[str, str]:
    """
    Convert probability to risk level and color.

    Args:
        probability: Flood susceptibility probability (0-1)

    Returns:
        Tuple of (risk_level, color_hex)

    Risk Thresholds:
        - < 0.25: low (green)
        - < 0.50: moderate (yellow)
        - < 0.75: high (orange)
        - >= 0.75: extreme (red)
    """
    if probability < 0.25:
        return "low", "#22c55e"       # green-500
    elif probability < 0.50:
        return "moderate", "#eab308"  # yellow-500
    elif probability < 0.75:
        return "high", "#f97316"      # orange-500
    else:
        return "extreme", "#ef4444"   # red-500


# Singleton instance for efficient model reuse
_model_instance: Optional[XGBoostHotspotModel] = None


def get_model() -> XGBoostHotspotModel:
    """
    Get the singleton model instance.

    Note: Model must be loaded separately via load_trained_model()
    before this instance can be used for predictions.
    """
    global _model_instance
    if _model_instance is None:
        _model_instance = XGBoostHotspotModel()
    return _model_instance


def load_trained_model(model_path: Path) -> XGBoostHotspotModel:
    """
    Load a trained model from disk and set as singleton.

    Args:
        model_path: Path to directory containing model files

    Returns:
        Loaded model instance
    """
    global _model_instance
    _model_instance = XGBoostHotspotModel()
    _model_instance.load(model_path)
    return _model_instance
