from fastapi import APIRouter, Depends, File, UploadFile, Form, HTTPException, status
from sqlalchemy.orm import Session
from src.database import get_db
from src import schemas, crud
from src.utils.images import save_upload
from io import BytesIO
import time

router = APIRouter(prefix="/reports", tags=["reports"])

@router.post("/", response_model=schemas.ReportRead, status_code=status.HTTP_201_CREATED)
async def create_report(
    title: str = Form(None),
    description: str = Form(None),
    lat: float = Form(...),
    lon: float = Form(...),
    image: UploadFile = File(None),
    db: Session = Depends(get_db)
):
    """
    Create a report with lat/lon and optional image upload.
    """
    image_path = None
    if image:
        # read contents and wrap in BytesIO so save_upload can read()
        contents = await image.read()
        image_obj = BytesIO(contents)
        filename = f"{int(time.time())}_{image.filename}"
        image_path = save_upload(image_obj, filename)

    # create report record
    try:
        report = crud.create_report(db, title=title, description=description, lon=lon, lat=lat, image_path=image_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create report: {e}")
    return report

@router.get("/", response_model=list[schemas.ReportRead])
def list_reports(limit: int = 100, db: Session = Depends(get_db)):
    """
    List recent reports (most recent first).
    """
    return crud.list_reports(db, limit=limit)
