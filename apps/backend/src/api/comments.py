"""
Comments API router for community feedback on flood reports.

Provides endpoints for:
- Getting comments on a report
- Adding comments to a report
- Deleting your own comments
"""
from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.orm import Session
from typing import List
from uuid import UUID
from datetime import datetime
import logging

from ..infrastructure.database import get_db
from ..infrastructure import models
from ..domain.models import CommentCreate, CommentResponse
from .deps import get_current_user

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/reports/{report_id}/comments", response_model=List[CommentResponse])
async def get_comments(
    report_id: UUID,
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db)
):
    """
    Get comments for a specific report.
    Returns comments ordered by creation time (oldest first).
    """
    try:
        # Verify report exists
        report = db.query(models.Report).filter(models.Report.id == report_id).first()
        if not report:
            raise HTTPException(status_code=404, detail="Report not found")

        # Get comments with user info
        comments = db.query(models.Comment, models.User).join(
            models.User, models.Comment.user_id == models.User.id
        ).filter(
            models.Comment.report_id == report_id
        ).order_by(
            models.Comment.created_at.asc()
        ).offset(offset).limit(limit).all()

        return [
            CommentResponse(
                id=comment.id,
                report_id=comment.report_id,
                user_id=comment.user_id,
                username=user.display_name or user.username,
                content=comment.content,
                created_at=comment.created_at
            )
            for comment, user in comments
        ]

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting comments: {e}")
        raise HTTPException(status_code=500, detail="Failed to get comments")


@router.post("/reports/{report_id}/comments", response_model=CommentResponse)
async def add_comment(
    report_id: UUID,
    comment: CommentCreate,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Add a comment to a report (requires authentication).
    Max 500 characters per comment.
    """
    try:
        # Verify report exists
        report = db.query(models.Report).filter(models.Report.id == report_id).first()
        if not report:
            raise HTTPException(status_code=404, detail="Report not found")

        # Check rate limiting (max 5 comments per minute per user)
        recent_comments = db.query(models.Comment).filter(
            models.Comment.user_id == current_user.id,
            models.Comment.created_at > datetime.utcnow().replace(second=0, microsecond=0)
        ).count()

        if recent_comments >= 5:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Max 5 comments per minute."
            )

        # Create comment
        new_comment = models.Comment(
            report_id=report_id,
            user_id=current_user.id,
            content=comment.content
        )
        db.add(new_comment)
        db.commit()
        db.refresh(new_comment)

        return CommentResponse(
            id=new_comment.id,
            report_id=new_comment.report_id,
            user_id=new_comment.user_id,
            username=current_user.display_name or current_user.username,
            content=new_comment.content,
            created_at=new_comment.created_at
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding comment: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to add comment")


@router.delete("/comments/{comment_id}")
async def delete_comment(
    comment_id: UUID,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Delete a comment (requires authentication).
    Only the comment author or an admin can delete a comment.
    """
    try:
        comment = db.query(models.Comment).filter(models.Comment.id == comment_id).first()
        if not comment:
            raise HTTPException(status_code=404, detail="Comment not found")

        # Check authorization: owner or admin
        is_owner = comment.user_id == current_user.id
        is_admin = current_user.role == 'admin'

        if not (is_owner or is_admin):
            raise HTTPException(
                status_code=403,
                detail="Not authorized to delete this comment"
            )

        db.delete(comment)
        db.commit()

        return {"message": "Comment deleted", "comment_id": comment_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting comment: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to delete comment")


@router.get("/reports/{report_id}/comments/count")
async def get_comment_count(
    report_id: UUID,
    db: Session = Depends(get_db)
):
    """
    Get the number of comments on a report.
    Lightweight endpoint for displaying counts without fetching all comments.
    """
    try:
        count = db.query(models.Comment).filter(
            models.Comment.report_id == report_id
        ).count()

        return {"report_id": report_id, "count": count}

    except Exception as e:
        logger.error(f"Error getting comment count: {e}")
        raise HTTPException(status_code=500, detail="Failed to get comment count")
