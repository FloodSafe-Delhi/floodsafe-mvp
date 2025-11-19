from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from sqlalchemy.orm import Session
from typing import Optional
from uuid import UUID
from PIL import Image
import io
import logging

from ..infrastructure.database import get_db
from ..infrastructure import models
from ..domain.models import ReportResponse
from ..core.utils import get_exif_data, get_lat_lon

router = APIRouter()
logger = logging.getLogger(__name__)

from typing import List

@router.get("/", response_model=List[ReportResponse])
def list_reports(db: Session = Depends(get_db)):
    """
    List all flood reports.
    """
    try:
        reports = db.query(models.Report).order_by(models.Report.timestamp.desc()).all()
        return reports
    except Exception as e:
        logger.error(f"Error listing reports: {e}")
        raise HTTPException(status_code=500, detail="Failed to list reports")

@router.post("/", response_model=ReportResponse)
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
            logger.error(f"Error processing image: {e}")
            raise HTTPException(status_code=400, detail=f"Image processing failed: {str(e)}")

    try:
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

        # Return proper DTO response
        return ReportResponse(
            id=new_report.id,
            description=new_report.description,
            latitude=latitude,
            longitude=longitude,
            media_url=new_report.media_url,
            verified=new_report.verified,
            verification_score=new_report.verification_score,
            upvotes=new_report.upvotes,
            timestamp=new_report.timestamp
        )
    except Exception as e:
        logger.error(f"Database error creating report: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to create report")

@router.post("/{report_id}/verify", response_model=ReportResponse)
def verify_report(report_id: UUID, db: Session = Depends(get_db)):
    """
    Verify a report and award points to the user.
    """
    try:
        report = db.query(models.Report).filter(models.Report.id == report_id).first()
        if not report:
            raise HTTPException(status_code=404, detail="Report not found")
            
        if report.verified:
            return ReportResponse(
                id=report.id,
                description=report.description,
                latitude=report.latitude,
                longitude=report.longitude,
                media_url=report.media_url,
                verified=report.verified,
                verification_score=report.verification_score,
                upvotes=report.upvotes,
                timestamp=report.timestamp
            )

        # Mark as verified
        report.verified = True
        report.verification_score += 10
        
        # Award points to user
        user = db.query(models.User).filter(models.User.id == report.user_id).first()
        if user:
            user.points += 10
            user.verified_reports_count += 1
            # Level up logic: 1 level per 100 points
            user.level = (user.points // 100) + 1
            
        db.commit()
        db.refresh(report)
        
        return ReportResponse(
            id=report.id,
            description=report.description,
            latitude=report.latitude,
            longitude=report.longitude,
            media_url=report.media_url,
            verified=report.verified,
            verification_score=report.verification_score,
            upvotes=report.upvotes,
            timestamp=report.timestamp
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error verifying report: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to verify report")
