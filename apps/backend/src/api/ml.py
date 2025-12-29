"""
ML Service Proxy API

Proxies ML classification requests from frontend to ML service.
This allows frontend to use a single backend URL (works in both local dev and Docker).

The frontend runs in the browser and cannot access Docker's internal network,
so we proxy through the backend which can reach ml-service:8002 in Docker.
"""

import logging
from typing import Optional

from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
import httpx

from ..core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()


class ClassificationResult(BaseModel):
    """Flood image classification result from ML service."""
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

    Proxies request to ML service for MobileNet classification.
    Uses low threshold (0.3) to minimize false negatives - safety first.

    Returns:
        ClassificationResult with classification, confidence, and review flags
    """
    if not settings.ML_SERVICE_ENABLED:
        raise HTTPException(
            status_code=503,
            detail="ML service is disabled. Set ML_SERVICE_ENABLED=true to enable."
        )

    # Validate file type
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {image.content_type}. Expected image/*"
        )

    try:
        content = await image.read()

        async with httpx.AsyncClient() as client:
            files = {
                "image": (
                    image.filename or "image.jpg",
                    content,
                    image.content_type or "image/jpeg"
                )
            }

            response = await client.post(
                f"{settings.ML_SERVICE_URL}/api/v1/classify-flood",
                files=files,
                timeout=15.0  # Slightly longer timeout for model inference
            )

            if response.status_code == 503:
                raise HTTPException(
                    status_code=503,
                    detail="ML classifier not loaded. Model weights may be missing."
                )

            if response.status_code != 200:
                logger.error(f"ML service returned {response.status_code}: {response.text}")
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"ML classification failed: {response.text}"
                )

            return response.json()

    except httpx.TimeoutException:
        logger.warning("ML classification timed out")
        raise HTTPException(
            status_code=504,
            detail="ML classification timed out. Please try again."
        )
    except httpx.ConnectError:
        logger.error(f"Cannot connect to ML service at {settings.ML_SERVICE_URL}")
        raise HTTPException(
            status_code=503,
            detail="ML service unavailable. Please try again later."
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"ML classification error: {e}")
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
    if not settings.ML_SERVICE_ENABLED:
        return ClassifierHealth(
            status="disabled",
            model_loaded=False,
            message="ML service is disabled in configuration"
        )

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{settings.ML_SERVICE_URL}/api/v1/classify-flood/health",
                timeout=5.0
            )

            if response.status_code == 200:
                return response.json()

            return ClassifierHealth(
                status="error",
                model_loaded=False,
                message=f"ML service returned {response.status_code}"
            )

    except httpx.ConnectError:
        return ClassifierHealth(
            status="unavailable",
            model_loaded=False,
            message=f"Cannot connect to ML service at {settings.ML_SERVICE_URL}"
        )
    except Exception as e:
        return ClassifierHealth(
            status="error",
            model_loaded=False,
            message=str(e)
        )
