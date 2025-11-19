from fastapi import FastAPI
from .infrastructure import models
from .infrastructure.database import engine

models.Base.metadata.create_all(bind=engine)

from .api import webhook, reports

app = FastAPI(title="FloodSafe API")

app.include_router(webhook.router, prefix="/api/webhooks", tags=["webhooks"])
app.include_router(reports.router, prefix="/api/reports", tags=["reports"])

@app.get("/health")
def health_check():
    return {"status": "healthy"}
