"""
FloodSafe ML Module - Embedded inference-only models.

This module contains lightweight ML models embedded directly in the backend
to eliminate the need for a separate ML service on resource-constrained
deployment platforms (e.g., Koyeb free tier).

Models:
- XGBoostHotspotModel: Waterlogging hotspot risk prediction (18-dim features)
- TFLiteFloodClassifier: Photo flood detection (MobileNet-based)

Note: These are INFERENCE-ONLY models. Training is done separately
using the full ml-service codebase.
"""

from .xgboost_hotspot import (
    XGBoostHotspotModel,
    get_risk_level,
    load_trained_model,
    FEATURE_NAMES,
)
from .tflite_classifier import (
    TFLiteFloodClassifier,
    get_classifier,
)
from .fhi_calculator import (
    FHICalculator,
    FHIResult,
    FHICalculationError,
    get_fhi_calculator,
    calculate_fhi_for_location,
)
from .hotspots_service import (
    HotspotsService,
    get_hotspots_service,
)

__all__ = [
    # XGBoost Hotspot Model
    "XGBoostHotspotModel",
    "get_risk_level",
    "load_trained_model",
    "FEATURE_NAMES",
    # TFLite Flood Classifier
    "TFLiteFloodClassifier",
    "get_classifier",
    # FHI Calculator
    "FHICalculator",
    "FHIResult",
    "FHICalculationError",
    "get_fhi_calculator",
    "calculate_fhi_for_location",
    # Hotspots Service
    "HotspotsService",
    "get_hotspots_service",
]
