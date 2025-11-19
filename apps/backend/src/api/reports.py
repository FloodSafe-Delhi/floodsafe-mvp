from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from sqlalchemy.orm import Session
from typing import Optional
from uuid import UUID
from PIL import Image
import io
import logging
import json

from ..infrastructure.database import get_db
from ..infrastructure import models
from ..domain.models import ReportResponse, Report as ReportDomain
from ..core.utils import get_exif_data, get_lat_lon
from ..domain.services.otp_service import get_otp_service
from ..domain.services.validation_service import ReportValidationService

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

        if not img_lat or not img_lng:
            raise HTTPException(
                status_code=400,
                detail="Photo must have GPS coordinates. Please enable location services and retake the photo."
            )

        media_metadata["gps"] = {"lat": img_lat, "lng": img_lng}

        # 3. Strict GPS validation: Photo GPS must match reported location within 100m
        # 100m ≈ 0.001 degrees at equator
        gps_tolerance = 0.001
        if abs(img_lat - latitude) > gps_tolerance or abs(img_lng - longitude) > gps_tolerance:
            raise HTTPException(
                status_code=400,
                detail=f"Photo location ({img_lat:.6f}, {img_lng:.6f}) does not match reported location ({latitude:.6f}, {longitude:.6f}). Please ensure you're taking a photo at the actual flood location."
            )

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
            vehicle_passability=vehicle_passability
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

            # Award points to user
            user = db.query(models.User).filter(models.User.id == user_id).first()
            if user:
                user.points += 15  # Bonus for IoT-validated report
                user.reports_count += 1
                user.verified_reports_count += 1
                user.level = (user.points // 100) + 1
        else:
            # Update counts even if not auto-verified
            user = db.query(models.User).filter(models.User.id == user_id).first()
            if user:
                user.reports_count += 1

        db.commit()
        db.refresh(new_report)

        logger.info(f"Report created: {new_report.id}, IoT score: {iot_score}, verified: {new_report.verified}")

        # Return response with new fields
        return ReportResponse(
            id=new_report.id,
            description=new_report.description,
            latitude=latitude,
            longitude=longitude,
            media_url=new_report.media_url,
            verified=new_report.verified,
            verification_score=new_report.verification_score,
            upvotes=new_report.upvotes,
            timestamp=new_report.timestamp,
            phone_verified=new_report.phone_verified,
            water_depth=new_report.water_depth,
            vehicle_passability=new_report.vehicle_passability,
            iot_validation_score=new_report.iot_validation_score
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
