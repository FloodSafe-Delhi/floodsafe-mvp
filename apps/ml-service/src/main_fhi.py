"""
FloodSafe ML Service - FHI-Only Deployment (Lightweight).

This is a minimal FastAPI service that ONLY provides:
- FHI (Flood Hazard Index) calculation using Open-Meteo API
- Hotspots endpoint with live FHI data

This version does NOT include:
- GEE (Google Earth Engine) integration
- Image classification (MobileNet/YOLO)
- Full ensemble ML models (LSTM/GNN/LightGBM)
- XGBoost model (optional - can be added if weights are bundled)

Total deployment size: ~50MB (vs 1.4GB+ for full service)
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="FloodSafe ML Service (FHI-Only)",
    description="Lightweight FHI calculation service using Open-Meteo API",
    version="1.0.0-fhi",
    docs_url="/api/v1/docs",
    redoc_url="/api/v1/redoc",
)

# CORS - allow all origins for now (ML service is internal)
CORS_ORIGINS = os.getenv("BACKEND_CORS_ORIGINS", "").split(",") if os.getenv("BACKEND_CORS_ORIGINS") else [
    "http://localhost:8000",
    "http://localhost:5175",
    "https://floodsafe-backend-floodsafe-dda84554.koyeb.app",
    "https://floodsafe-mvp-frontend.vercel.app",
]
CORS_ORIGINS = [origin.strip() for origin in CORS_ORIGINS if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS + ["*"],  # Be permissive for internal service
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import hotspots router (only API we need)
from .api.hotspots import router as hotspots_router, initialize_hotspots_router

# Include hotspots router
app.include_router(
    hotspots_router,
    prefix="/api/v1/hotspots",
    tags=["hotspots"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("=" * 60)
    logger.info("Starting FloodSafe ML Service (FHI-Only Mode)...")
    logger.info("=" * 60)

    # Initialize hotspots service (loads JSON data, no ML model needed)
    try:
        logger.info("Initializing hotspots service...")
        initialize_hotspots_router()

        # Import to check loaded count
        from .api import hotspots
        hotspot_count = len(hotspots.hotspots_data)
        logger.info(f"  Hotspots loaded: {hotspot_count}")
        logger.info(f"  FHI calculator: using Open-Meteo API (free, no auth)")
    except Exception as e:
        logger.error(f"[ERROR] Hotspots initialization failed: {e}")
        raise

    # Startup summary
    logger.info("=" * 60)
    logger.info("ML Service (FHI-Only) startup complete")
    logger.info(f"  Hotspots: {hotspot_count} locations")
    logger.info(f"  FHI: Open-Meteo API (live weather)")
    logger.info(f"  GEE: disabled")
    logger.info(f"  ML Models: disabled (FHI-only mode)")
    logger.info(f"  Docs: /api/v1/docs")
    logger.info("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down ML Service (FHI-Only)...")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "FloodSafe ML Service (FHI-Only)",
        "version": "1.0.0-fhi",
        "status": "running",
        "mode": "fhi-only",
        "docs": "/api/v1/docs",
        "health": "/health",
        "features": {
            "fhi": True,
            "hotspots": True,
            "gee": False,
            "ml_models": False,
            "image_classification": False,
        },
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    from .api import hotspots

    return {
        "status": "healthy",
        "service": "FloodSafe ML Service (FHI-Only)",
        "mode": "fhi-only",
        "hotspots_loaded": len(hotspots.hotspots_data) > 0,
        "hotspots_count": len(hotspots.hotspots_data),
        "fhi_source": "open-meteo",
    }


if __name__ == "__main__":
    import uvicorn

    # Use PORT env var for Koyeb (auto-set) or FASTAPI_PORT or default 8002
    api_port = int(os.getenv("PORT", os.getenv("FASTAPI_PORT", "8002")))

    uvicorn.run(
        "src.main_fhi:app",
        host="0.0.0.0",
        port=api_port,
        log_level="info",
    )
