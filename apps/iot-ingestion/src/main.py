import asyncio
import uvicorn
from fastapi import FastAPI

app = FastAPI(title="FloodSafe IoT Ingestion")

@app.post("/ingest")
async def ingest_data(data: dict):
    print(f"Received data: {data}")
    # TODO: Push to Queue or Database
    return {"status": "received"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
