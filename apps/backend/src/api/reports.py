from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, Query
from sqlalchemy.orm import Session
from typing import Optional, List
from uuid import UUID
from PIL import Image
import io
import logging
import json

from ..infrastructure.database import get_db
from ..infrastructure import models
from ..domain.models import ReportResponse, Report as ReportDomain, UserResponse
from ..domain.reputation_models import ReportVerificationRequest
from ..domain.services.reputation_service import ReputationService
from ..core.utils import get_exif_data, get_lat_lon
from ..domain.services.otp_service import get_otp_service
from ..domain.services.validation_service import ReportValidationService
from geoalchemy2.functions import ST_DWithin, ST_MakePoint

router = APIRouter()
logger = logging.getLogger(__name__)

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

@router.get("/location/details", response_model=dict)
def get_location_details(
    latitude: float = Query(..., ge=-90, le=90),
    longitude: float = Query(..., ge=-180, le=180),
    radius_meters: float = Query(500, gt=0, le=5000),
    db: Session = Depends(get_db)
):
    """
    Get all reports and user information at a specific location.
    This is used when user clicks "Locate" on an alert to see details.

    Returns:
    - List of reports at this location
    - Users who reported (with their report counts)
    - Count of total reports
    """
    try:
        # Create a point for the query location
        query_point = ST_MakePoint(longitude, latitude)

        # Find all reports within radius
        nearby_reports = db.query(models.Report).filter(
            ST_DWithin(
                models.Report.location,
                query_point,
                radius_meters,
                True  # Use spheroid for accurate distance
            )
        ).order_by(models.Report.timestamp.desc()).all()

        # Get unique user IDs from these reports
        user_ids = list(set([r.user_id for r in nearby_reports]))

        # Get user details
        users = db.query(models.User).filter(models.User.id.in_(user_ids)).all() if user_ids else []

        # Build response
        report_details = []
        for report in nearby_reports:
            report_details.append({
                "id": str(report.id),
                "description": report.description,
                "latitude": report.latitude,
                "longitude": report.longitude,
                "verified": report.verified,
                "upvotes": report.upvotes,
                "timestamp": report.timestamp.isoformat(),
                "user_id": str(report.user_id)
            })

        user_details = []
        for user in users:
            user_details.append({
                "id": str(user.id),
                "username": user.username,
                "reports_count": user.reports_count,
                "verified_reports_count": user.verified_reports_count,
                "level": user.level
            })

        return {
            "location": {
                "latitude": latitude,
                "longitude": longitude,
                "radius_meters": radius_meters
            },
            "total_reports": len(nearby_reports),
            "reports": report_details,
            "reporters": user_details
        }

    except Exception as e:
        logger.error(f"Error getting location details: {e}")
        raise HTTPException(status_code=500, detail="Failed to get location details")

@router.post("/", response_model=ReportResponse)
async def create_report(
    user_id: UUID = Form(...),
    description: str = Form(...),
    latitude: float = Form(...),
    longitude: float = Form(...),
    phone_number: str = Form(...),
    phone_verification_token: str = Form(...),
    water_depth: Optional[str] = Form(None),
    vehicle_passability: Optional[str] = Form(None),
    image: UploadFile = File(...),  # MANDATORY - changed from File(None)
    db: Session = Depends(get_db)
):
    """
    Create a new flood report with community verification.

    Required fields:
    - Geotagged photo (MANDATORY)
    - Phone number with OTP verification
    - Location coordinates

    Validation:
    1. Verify phone number via OTP token
    2. Extract GPS from photo EXIF
    3. Validate photo GPS matches reported location (±100m)
    4. Cross-reference with IoT sensor data
    5. Calculate validation score
    """
    # 1. Verify phone number via OTP token
    otp_service = get_otp_service()
    if not otp_service.verify_token(phone_number, phone_verification_token):
        raise HTTPException(
            status_code=401,
            detail="Invalid phone verification token. Please verify your phone number."
        )

    media_url = None
    media_metadata = {}

    # 2. Process mandatory image and extract EXIF GPS
    content = await image.read()
    try:
        img = Image.open(io.BytesIO(content))
        exif_data = get_exif_data(img)
        img_lat, img_lng = get_lat_lon(exif_data)

        # Initialize location_verified flag
        location_verified = True

        if not img_lat or not img_lng:
            # No GPS in photo - flag as not location verified (don't block)
            location_verified = False
            logger.warning(f"Report photo has no GPS coordinates")
        else:
            media_metadata["gps"] = {"lat": img_lat, "lng": img_lng}

            # 3. GPS validation: Check if photo GPS matches reported location within 100m
            # 100m ≈ 0.001 degrees at equator
            gps_tolerance = 0.001
            if abs(img_lat - latitude) > gps_tolerance or abs(img_lng - longitude) > gps_tolerance:
                # GPS mismatch - flag as not location verified (don't block, allow with warning)
                location_verified = False
                logger.warning(f"Report photo GPS ({img_lat:.6f}, {img_lng:.6f}) doesn't match reported location ({latitude:.6f}, {longitude:.6f})")

        # TODO: Upload to S3/Blob Storage and get URL
        # media_url = s3_upload(content)
        media_url = f"https://mock-storage.com/{image.filename}"

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=400, detail=f"Image processing failed: {str(e)}")

    try:
        # 4. Create report with new fields
        new_report = models.Report(
            user_id=user_id,
            description=description,
            location=f"POINT({longitude} {latitude})",  # PostGIS Point
            media_url=media_url,
            media_type="image",
            media_metadata=json.dumps(media_metadata),
            phone_number=phone_number,
            phone_verified=True,  # Verified via OTP
            water_depth=water_depth,
            vehicle_passability=vehicle_passability,
            location_verified=location_verified  # Photo GPS matches reported location
        )

        db.add(new_report)
        db.flush()  # Flush to get ID, but don't commit yet

        # 5. Validate against IoT sensors
        validation_service = ReportValidationService(db)

        # Create domain model for validation
        report_domain = ReportDomain(
            id=new_report.id,
            user_id=user_id,
            description=description,
            location_lat=latitude,
            location_lng=longitude,
            water_depth=water_depth,
            vehicle_passability=vehicle_passability,
            media_url=media_url,
            phone_number=phone_number,
            phone_verified=True
        )

        iot_score = validation_service.validate_report(report_domain)

        # Get nearby sensors for reference
        nearby_sensors = validation_service._find_nearby_sensors(latitude, longitude, 1000)
        nearby_sensor_ids = [str(s.id) for s in nearby_sensors]

        # Update report with validation results
        new_report.iot_validation_score = iot_score
        new_report.nearby_sensor_ids = json.dumps(nearby_sensor_ids)

        # 6. Auto-verify if IoT validation score is high
        if iot_score >= 80:
            new_report.verified = True
            new_report.verification_score += 20

            # Award points to user with reputation system
            user = db.query(models.User).filter(models.User.id == user_id).first()
            if user:
                user.points += 15  # Bonus for IoT-validated report
                user.reports_count += 1
                user.verified_reports_count += 1
                user.level = (user.points // 100) + 1
        else:
            # Update counts and give base points even if not auto-verified
            user = db.query(models.User).filter(models.User.id == user_id).first()
            if user:
                user.reports_count += 1
                user.points += 5  # Base submission points
                user.level = (user.points // 100) + 1

        db.commit()
        db.refresh(new_report)

        db.commit()
        db.refresh(new_report)

        logger.info(f"Report created: {new_report.id}, IoT score: {iot_score}, verified: {new_report.verified}")

        # Update streak (using reputation service)
        reputation_service = ReputationService(db)
        streak_bonus = reputation_service.update_streak(user_id)

        if streak_bonus:
            logger.info(f"User {user_id} earned streak bonus: {streak_bonus} points")

        # Return response with combined fields from both features
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
            timestamp=new_report.timestamp,
            phone_verified=new_report.phone_verified,
            water_depth=new_report.water_depth,
            vehicle_passability=new_report.vehicle_passability,
            iot_validation_score=new_report.iot_validation_score,
            location_verified=new_report.location_verified
        )
    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        logger.error(f"Database error creating report: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to create report: {str(e)}")

@router.get("/hyperlocal")
def get_hyperlocal_status(
    lat: float,
    lng: float,
    radius: int = 500,  # meters
    db: Session = Depends(get_db)
):
    """
    Get hyperlocal area status for a specific location.

    Returns:
    - All reports within radius
    - Aggregate status (safe/caution/warning/critical)
    - Area summary statistics
    - Sensor data summary
    """
    from datetime import timedelta
    from sqlalchemy import text

    try:
        # 1. Find reports in radius using PostGIS
        query = text("""
            SELECT
                id, description, verified, water_depth, vehicle_passability,
                iot_validation_score, timestamp,
                ST_X(location::geometry) as longitude,
                ST_Y(location::geometry) as latitude
            FROM reports
            WHERE ST_DWithin(
                location::geography,
                ST_SetSRID(ST_MakePoint(:lng, :lat), 4326)::geography,
                :radius
            )
            AND timestamp > NOW() - INTERVAL '24 hours'
            ORDER BY timestamp DESC
        """)

        result = db.execute(query, {'lat': lat, 'lng': lng, 'radius': radius})
        reports = []

        verified_count = 0
        water_depths = []

        for row in result:
            report_dict = {
                'id': str(row[0]),
                'description': row[1],
                'verified': row[2],
                'water_depth': row[3],
                'vehicle_passability': row[4],
                'iot_validation_score': row[5],
                'timestamp': row[6].isoformat() if row[6] else None,
                'longitude': float(row[7]) if row[7] else None,
                'latitude': float(row[8]) if row[8] else None
            }
            reports.append(report_dict)

            if row[2]:  # verified
                verified_count += 1
            if row[3]:  # water_depth
                water_depths.append(row[3])

        # 2. Get sensor data summary
        validation_service = ReportValidationService(db)
        sensor_summary = validation_service.get_nearby_sensor_summary(lat, lng, radius)

        # 3. Calculate aggregate status
        total_reports = len(reports)
        avg_validation_score = sum(r['iot_validation_score'] or 0 for r in reports) / total_reports if total_reports > 0 else 0

        # Determine status based on multiple factors
        if total_reports == 0:
            status = sensor_summary.get('status', 'unknown')
        else:
            # Count critical indicators
            critical_count = sum(1 for r in reports if r['water_depth'] in ['waist', 'impassable'])
            high_count = sum(1 for r in reports if r['water_depth'] == 'knee')

            if critical_count > 0 or sensor_summary.get('status') == 'critical':
                status = 'critical'
            elif high_count > 1 or sensor_summary.get('status') == 'warning':
                status = 'warning'
            elif total_reports > 2 or sensor_summary.get('status') == 'caution':
                status = 'caution'
            else:
                status = 'safe'

        # 4. Calculate average water depth
        depth_order = {'ankle': 1, 'knee': 2, 'waist': 3, 'impassable': 4}
        avg_depth = None
        if water_depths:
            avg_depth_value = sum(depth_order.get(d, 0) for d in water_depths) / len(water_depths)
            if avg_depth_value >= 3.5:
                avg_depth = 'impassable'
            elif avg_depth_value >= 2.5:
                avg_depth = 'waist'
            elif avg_depth_value >= 1.5:
                avg_depth = 'knee'
            else:
                avg_depth = 'ankle'

        return {
            'reports': reports,
            'status': status,
            'area_summary': {
                'total_reports': total_reports,
                'verified_reports': verified_count,
                'avg_water_depth': avg_depth,
                'avg_validation_score': round(avg_validation_score, 1),
                'last_updated': reports[0]['timestamp'] if reports else None
            },
            'sensor_summary': sensor_summary
        }

    except Exception as e:
        logger.error(f"Error getting hyperlocal status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get hyperlocal status: {str(e)}")


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
