from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from sqlalchemy.orm import Session
from typing import Optional
from uuid import UUID
from PIL import Image
import io
import logging

from ..infrastructure.database import get_db
from ..infrastructure import models
from ..domain.models import ReportResponse, UserResponse
from ..domain.reputation_models import ReportVerificationRequest
from ..domain.services.reputation_service import ReputationService
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

        # Update user's report count
        user = db.query(models.User).filter(models.User.id == user_id).first()
        if user:
            user.reports_count += 1

            # Award submission points
            user.points += 5
            user.level = (user.points // 100) + 1

        db.commit()
        db.refresh(new_report)

        # Update streak (after commit)
        reputation_service = ReputationService(db)
        streak_bonus = reputation_service.update_streak(user_id)

        if streak_bonus:
            logger.info(f"User {user_id} earned streak bonus: {streak_bonus} points")

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
            downvotes=new_report.downvotes,
            quality_score=new_report.quality_score,
            verified_at=new_report.verified_at,
            timestamp=new_report.timestamp
        )
    except Exception as e:
        logger.error(f"Database error creating report: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to create report")

@router.post("/{report_id}/verify")
def verify_report(
    report_id: UUID,
    verification: ReportVerificationRequest,
    db: Session = Depends(get_db)
):
    """
    Verify or reject a report with quality scoring.

    Uses the reputation system to:
    - Calculate quality score
    - Award points based on quality
    - Update user reputation
    - Check for badges
    - Log history
    """
    try:
        report = db.query(models.Report).filter(models.Report.id == report_id).first()
        if not report:
            raise HTTPException(status_code=404, detail="Report not found")

        if report.verified:
            # Already verified
            return {
                'message': 'Report already verified',
                'report_id': report_id,
                'verified': True,
                'quality_score': report.quality_score
            }

        # Process verification through reputation service
        reputation_service = ReputationService(db)
        result = reputation_service.process_report_verification(
            report_id=report_id,
            verified=verification.verified,
            quality_score=verification.quality_score
        )

        return {
            'message': 'Report verified' if verification.verified else 'Report rejected',
            'report_id': report_id,
            'verified': verification.verified,
            **result
        }

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error verifying report: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to verify report")


@router.post("/{report_id}/upvote")
def upvote_report(report_id: UUID, db: Session = Depends(get_db)):
    """
    Upvote a report.
    Awards small bonus to report owner.
    """
    try:
        report = db.query(models.Report).filter(models.Report.id == report_id).first()
        if not report:
            raise HTTPException(status_code=404, detail="Report not found")

        # Increment upvotes
        report.upvotes += 1
        db.commit()

        # Award bonus to report owner
        reputation_service = ReputationService(db)
        result = reputation_service.process_report_upvote(report_id)

        return {
            'message': 'Report upvoted',
            'report_id': report_id,
            'upvotes': report.upvotes,
            **result
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error upvoting report: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to upvote report")


@router.post("/{report_id}/downvote")
def downvote_report(report_id: UUID, db: Session = Depends(get_db)):
    """
    Downvote a report.
    Used to flag potentially false reports.
    """
    try:
        report = db.query(models.Report).filter(models.Report.id == report_id).first()
        if not report:
            raise HTTPException(status_code=404, detail="Report not found")

        # Increment downvotes
        report.downvotes += 1
        db.commit()

        return {
            'message': 'Report downvoted',
            'report_id': report_id,
            'downvotes': report.downvotes
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downvoting report: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to downvote report")
