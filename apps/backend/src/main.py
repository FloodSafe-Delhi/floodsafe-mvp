from fastapi import FastAPI
from src.database import engine
from src.models import user_report
from src.api import auth, reports, webhook

app = FastAPI(title="FloodSafe Backend API")

@app.on_event("startup")
def startup():
    user_report.Base.metadata.create_all(bind=engine)

app.include_router(auth.router)
app.include_router(reports.router)
app.include_router(webhook.router)

@app.get("/health")
def health():
    return {"status":"ok"}
