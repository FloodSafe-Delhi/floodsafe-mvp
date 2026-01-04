"""
ML API - Embedded flood image classification.

Uses TFLite-based MobileNet classifier for flood detection.
No external ML service dependency - runs entirely in backend.

Safety-first approach: Low threshold (0.3) to minimize false negatives.
Better to flag a non-flood image for review than miss a real flood.
"""

import logging
from typing import Optional
from io import BytesIO

from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel

from ..core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()

# Lazy-loaded classifier instance
_classifier = None


def _get_classifier():
    """Get or create the TFLite classifier instance."""
    global _classifier

    if _classifier is None:
        try:
            from ..domain.ml.tflite_classifier import get_classifier
            _classifier = get_classifier()
            logger.info("TFLite flood classifier loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load TFLite classifier: {e}")
            raise RuntimeError(f"Classifier not available: {e}")

    return _classifier


class ClassificationResult(BaseModel):
    """Flood image classification result."""
    classification: str  # "flood" or "no_flood"
    confidence: float  # 0.0-1.0
    flood_probability: float  # 0.0-1.0
    is_flood: bool
    needs_review: bool  # True if confidence is uncertain (0.3-0.7)
    verification_score: int  # 0-100 for report credibility
    probabilities: dict  # {"flood": 0.92, "no_flood": 0.08}


class ClassifierHealth(BaseModel):
    """Health status of the ML classifier."""
    status: str
    model_loaded: bool
    model_path: Optional[str] = None
    threshold: Optional[float] = None
    message: Optional[str] = None


@router.post("/classify-flood", response_model=ClassificationResult)
async def classify_flood_image(
    image: UploadFile = File(..., description="Image file to classify (JPEG, PNG)")
):
    """
    Classify uploaded image as flood or not flood.

    Uses embedded TFLite MobileNet classifier.
    Low threshold (0.3) to minimize false negatives - safety first.

    Returns:
        ClassificationResult with classification, confidence, and review flags
    """
    if not settings.ML_ENABLED:
        raise HTTPException(
            status_code=503,
            detail="ML is disabled. Set ML_ENABLED=true to enable."
        )

    # Validate file type
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {image.content_type}. Expected image/*"
        )

    try:
        # Get classifier
        classifier = _get_classifier()

        # Read image content
        content = await image.read()

        # Classify image
        result = classifier.predict(BytesIO(content))

        # Calculate verification score (0-100)
        # Higher confidence = higher verification score
        if result["is_flood"]:
            # For floods, high probability = high score
            verification_score = int(result["flood_probability"] * 100)
        else:
            # For non-floods, high no_flood probability = high score
            verification_score = int(result["probabilities"]["no_flood"] * 100)

        # Reduce score if needs review (uncertain classification)
        if result["needs_review"]:
            verification_score = max(30, verification_score - 20)

        return ClassificationResult(
            classification=result["classification"],
            confidence=result["confidence"],
            flood_probability=result["flood_probability"],
            is_flood=result["is_flood"],
            needs_review=result["needs_review"],
            verification_score=verification_score,
            probabilities=result["probabilities"],
        )

    except RuntimeError as e:
        # Classifier not loaded
        logger.error(f"Classifier error: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"ML classifier not available: {str(e)}"
        )
    except ValueError as e:
        # Image processing error
        logger.error(f"Image processing error: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Could not process image: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Classification error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Classification failed: {str(e)}"
        )


@router.get("/classify-flood/health", response_model=ClassifierHealth)
async def classifier_health():
    """
    Check ML classifier health status.

    Returns whether the classifier model is loaded and ready.
    """
    if not settings.ML_ENABLED:
        return ClassifierHealth(
            status="disabled",
            model_loaded=False,
            message="ML is disabled in configuration"
        )

    try:
        classifier = _get_classifier()

        info = classifier.get_model_info()
        return ClassifierHealth(
            status="healthy" if classifier.is_loaded else "not_loaded",
            model_loaded=classifier.is_loaded,
            model_path=info.get("model_path"),
            threshold=info.get("threshold"),
            message=f"TFLite classifier ready ({info.get('architecture', 'unknown')})"
        )

    except RuntimeError as e:
        return ClassifierHealth(
            status="error",
            model_loaded=False,
            message=str(e)
        )
    except Exception as e:
        return ClassifierHealth(
            status="error",
            model_loaded=False,
            message=f"Unexpected error: {str(e)}"
        )
