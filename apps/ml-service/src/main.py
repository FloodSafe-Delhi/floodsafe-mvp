"""
FloodSafe ML Service - Main Application.

FastAPI service for flood prediction using ensemble ML models.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
from pathlib import Path

from .core.config import settings
from .api import predictions, hotspots
from .models.ensemble import create_default_ensemble
from .features.extractor import FeatureExtractor
from .data.gee_client import gee_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url=f"{settings.API_V1_STR}/docs",
    redoc_url=f"{settings.API_V1_STR}/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(
    predictions.router,
    prefix=f"{settings.API_V1_STR}/predictions",
    tags=["predictions"],
)
app.include_router(
    hotspots.router,
    prefix=f"{settings.API_V1_STR}/hotspots",
    tags=["hotspots"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("=" * 60)
    logger.info(f"Starting {settings.PROJECT_NAME}...")
    logger.info("=" * 60)

    # Initialize GEE
    try:
        logger.info("Initializing Google Earth Engine...")
        gee_client.initialize()
        logger.info("✓ GEE initialized successfully")
    except Exception as e:
        logger.warning(f"✗ GEE initialization failed: {e}")
        logger.warning("  Service will continue but predictions may fail")

    # Initialize feature extractor
    try:
        logger.info("Initializing feature extractor...")
        predictions.feature_extractor = FeatureExtractor(lazy_load=True)
        logger.info("✓ Feature extractor ready")
    except Exception as e:
        logger.error(f"✗ Feature extractor initialization failed: {e}")

    # Load or create ensemble model
    try:
        model_path = Path(settings.MODEL_CACHE_DIR) / "ensemble"

        if model_path.exists():
            logger.info(f"Loading pre-trained ensemble from {model_path}...")
            from .models.ensemble import EnsembleFloodModel

            predictions.ensemble_model = EnsembleFloodModel().load(model_path)
            logger.info("✓ Ensemble model loaded from disk")
        else:
            logger.info("No pre-trained model found, creating default ensemble...")
            predictions.ensemble_model = create_default_ensemble()
            logger.info("✓ Default ensemble created (not trained)")
            logger.warning(
                "  Model needs training before making predictions."
                " Use /train endpoint or train offline."
            )

    except Exception as e:
        logger.error(f"✗ Model initialization failed: {e}")
        predictions.ensemble_model = None

    # Initialize hotspots service
    try:
        logger.info("Initializing hotspots service...")
        hotspots.initialize_hotspots_router()
        logger.info(f"  Hotspots loaded: {len(hotspots.hotspots_data)}")
        logger.info(f"  Hotspot model: {'trained' if hotspots.hotspot_model and hotspots.hotspot_model.is_trained else 'not available'}")
    except Exception as e:
        logger.error(f"Hotspots initialization failed: {e}")

    # Load pre-computed grid predictions cache (FAST forecasts)
    try:
        logger.info("Loading grid predictions cache...")
        if predictions.load_grid_predictions_cache():
            cache_points = len(predictions.grid_predictions_cache.get("features", []))
            logger.info(f"✓ Grid predictions cache loaded: {cache_points} points")
        else:
            logger.warning("✗ No grid predictions cache - /forecast-grid will be slow")
    except Exception as e:
        logger.error(f"Grid cache loading failed: {e}")

    logger.info("=" * 60)
    logger.info("ML Service startup complete")
    logger.info(f"  Model loaded: {predictions.ensemble_model is not None}")
    logger.info(f"  Model trained: {predictions.ensemble_model.is_trained if predictions.ensemble_model else False}")
    logger.info(f"  Hotspots: {len(hotspots.hotspots_data)} locations")
    logger.info(f"  Grid cache: {len(predictions.grid_predictions_cache.get('features', [])) if predictions.grid_predictions_cache else 0} points")
    logger.info(f"  GEE available: {gee_client._initialized}")
    logger.info(f"  Docs: {settings.API_V1_STR}/docs")
    logger.info("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down ML Service...")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": settings.PROJECT_NAME,
        "version": "1.0.0",
        "status": "running",
        "docs": f"{settings.API_V1_STR}/docs",
        "health": "/api/v1/predictions/health",
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8002,
        reload=True,
        log_level="info",
    )
