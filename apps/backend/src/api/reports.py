from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from sqlalchemy.orm import Session
from typing import Optional
from uuid import UUID
from PIL import Image
import io

from ..infrastructure.database import get_db
from ..infrastructure import models
from ..core.utils import get_exif_data, get_lat_lon

router = APIRouter()

@router.post("/")
async def create_report(
    user_id: UUID = Form(...),
    description: str = Form(...),
    latitude: float = Form(...),
    longitude: float = Form(...),
    image: UploadFile = File(None),
    db: Session = Depends(get_db)
):
    """
    Create a new flood report.
    If an image is provided, it verifies the geotag matches the reported location (within tolerance).
    """
    media_url = None
    media_metadata = {}
    
    if image:
        # Read image file
        content = await image.read()
        try:
            img = Image.open(io.BytesIO(content))
            exif_data = get_exif_data(img)
            img_lat, img_lng = get_lat_lon(exif_data)
            
            if img_lat and img_lng:
                media_metadata["gps"] = {"lat": img_lat, "lng": img_lng}
                
                # Simple verification: Check if image location is close to reported location
                # Tolerance: ~1km (roughly 0.01 degrees)
                if abs(img_lat - latitude) > 0.01 or abs(img_lng - longitude) > 0.01:
                     # We don't block it, but we flag it as unverified or suspicious
                     # For MVP, we just note it in metadata
                     media_metadata["location_mismatch"] = True
            
            # TODO: Upload to S3/Blob Storage and get URL
            # media_url = s3_upload(content)
            media_url = f"https://mock-storage.com/{image.filename}" 
            
        except Exception as e:
            print(f"Error processing image: {e}")
            # Proceed without image processing if it fails (e.g. not a valid image)

    new_report = models.Report(
        user_id=user_id,
        description=description,
        # PostGIS Point: POINT(lng lat)
        location=f"POINT({longitude} {latitude})", 
        media_url=media_url,
        media_type="image" if image else "text",
        media_metadata=str(media_metadata)
    )
    
    db.add(new_report)
    db.commit()
    db.refresh(new_report)
    
    return {"status": "success", "report_id": new_report.id, "verified_location": "gps" in media_metadata}
