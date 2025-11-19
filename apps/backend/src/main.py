from fastapi import FastAPI
from .infrastructure import models
from .infrastructure.database import engine

models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="FloodSafe API")

@app.get("/health")
def health_check():
    return {"status": "healthy"}
