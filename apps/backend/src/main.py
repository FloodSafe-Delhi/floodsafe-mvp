from fastapi import FastAPI
from .infrastructure import models
from .infrastructure.database import engine

models.Base.metadata.create_all(bind=engine)

from .api import webhook, reports, users, sensors, otp

from fastapi.middleware.cors import CORSMiddleware
from .core.config import settings

app = FastAPI(title=settings.PROJECT_NAME)

# Set all CORS enabled origins
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.BACKEND_CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.include_router(webhook.router, prefix="/api/webhooks", tags=["webhooks"])
app.include_router(reports.router, prefix="/api/reports", tags=["reports"])
app.include_router(users.router, prefix="/api/users", tags=["users"])
app.include_router(sensors.router, prefix="/api/sensors", tags=["sensors"])
app.include_router(otp.router, prefix="/api", tags=["otp"])

@app.get("/health")
def health_check():
    return {"status": "healthy"}
