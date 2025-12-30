"""
Supabase Storage service for report photos.

This module provides a clean abstraction for uploading images to Supabase Storage.
Raises explicit errors if storage is not configured - no silent fallbacks.

Usage:
    from ..infrastructure.storage import get_storage_service, StorageError, StorageNotConfiguredError

    storage = get_storage_service()
    public_url, path = await storage.upload_image(content, filename, content_type, user_id)
"""
import httpx
import uuid
import logging
from typing import Optional, Tuple
from datetime import datetime

from ..core.config import settings

logger = logging.getLogger(__name__)


class StorageError(Exception):
    """Raised when storage operation fails."""
    pass


class StorageNotConfiguredError(StorageError):
    """Raised when storage is not configured. This is a deployment issue, not a user error."""
    pass


class SupabaseStorageService:
    """
    Handles file uploads to Supabase Storage.

    Features:
    - Uploads images to a public bucket for easy access
    - Generates unique paths to avoid collisions
    - Raises explicit errors if not configured (no silent fallbacks)
    - Supports image deletion for cleanup
    """

    def __init__(self):
        self.url = settings.SUPABASE_URL
        self.key = settings.SUPABASE_SERVICE_KEY
        self.bucket = settings.SUPABASE_STORAGE_BUCKET

    @property
    def is_configured(self) -> bool:
        """Check if Supabase storage is properly configured."""
        return bool(self.url and self.key)

    def _get_headers(self) -> dict:
        """Get headers for Supabase API requests."""
        return {
            "Authorization": f"Bearer {self.key}",
            "apikey": self.key
        }

    def _generate_path(self, filename: str, user_id: str) -> str:
        """
        Generate unique storage path for file.

        Format: reports/{user_id}/{timestamp}_{unique_id}_{safe_filename}
        Example: reports/abc123/20240101_120000_a1b2c3d4_flood_photo.jpg
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:8]
        # Sanitize filename - keep only alphanumeric, dots, dashes, underscores
        safe_filename = "".join(c if c.isalnum() or c in ".-_" else "_" for c in filename)
        # Limit filename length
        if len(safe_filename) > 50:
            ext_idx = safe_filename.rfind(".")
            if ext_idx > 0:
                ext = safe_filename[ext_idx:]
                safe_filename = safe_filename[:50-len(ext)] + ext
            else:
                safe_filename = safe_filename[:50]
        return f"reports/{user_id}/{timestamp}_{unique_id}_{safe_filename}"

    async def upload_image(
        self,
        content: bytes,
        filename: str,
        content_type: str,
        user_id: str
    ) -> Tuple[str, str]:
        """
        Upload image to Supabase Storage.

        Args:
            content: Image bytes
            filename: Original filename
            content_type: MIME type (e.g., image/jpeg)
            user_id: User ID for path organization

        Returns:
            Tuple of (public_url, storage_path)

        Raises:
            StorageNotConfiguredError: If Supabase storage is not configured
            StorageError: If upload fails
        """
        if not self.is_configured:
            logger.error("Supabase storage not configured! Set SUPABASE_URL and SUPABASE_SERVICE_KEY env vars.")
            raise StorageNotConfiguredError(
                "Photo storage not configured. Please contact support - this is a deployment issue."
            )

        path = self._generate_path(filename, user_id)
        upload_url = f"{self.url}/storage/v1/object/{self.bucket}/{path}"

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    upload_url,
                    content=content,
                    headers={
                        **self._get_headers(),
                        "Content-Type": content_type,
                        "x-upsert": "true"  # Overwrite if exists
                    },
                    timeout=30.0
                )

                if response.status_code not in (200, 201):
                    error_detail = response.text[:500] if response.text else "Unknown error"
                    logger.error(f"Supabase upload failed: {response.status_code} - {error_detail}")
                    raise StorageError(f"Upload failed with status {response.status_code}: {error_detail}")

                # Construct public URL for the uploaded file
                public_url = f"{self.url}/storage/v1/object/public/{self.bucket}/{path}"
                logger.info(f"Successfully uploaded image to Supabase: {path}")
                return public_url, path

        except httpx.TimeoutException:
            logger.error(f"Supabase upload timed out for {filename}")
            raise StorageError("Upload timed out after 30 seconds")
        except httpx.RequestError as e:
            logger.error(f"Supabase upload request failed: {e}")
            raise StorageError(f"Request failed: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during Supabase upload: {e}")
            raise StorageError(f"Upload failed: {str(e)}")

    async def delete_image(self, path: str) -> bool:
        """
        Delete image from Supabase Storage.

        Args:
            path: Storage path returned from upload_image

        Returns:
            True if deleted successfully, False otherwise
        """
        if not self.is_configured:
            logger.warning("Supabase not configured, cannot delete image")
            return False

        delete_url = f"{self.url}/storage/v1/object/{self.bucket}/{path}"

        try:
            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    delete_url,
                    headers=self._get_headers(),
                    timeout=10.0
                )
                success = response.status_code in (200, 204, 404)  # 404 is ok (already deleted)
                if success:
                    logger.info(f"Deleted image from Supabase: {path}")
                else:
                    logger.warning(f"Failed to delete {path}: {response.status_code}")
                return success
        except Exception as e:
            logger.warning(f"Error deleting image {path}: {e}")
            return False

    async def get_public_url(self, path: str) -> str:
        """
        Get public URL for an existing storage path.

        Args:
            path: Storage path

        Returns:
            Public URL string
        """
        if not self.is_configured:
            raise StorageNotConfiguredError("Storage not configured, cannot generate URL")
        return f"{self.url}/storage/v1/object/public/{self.bucket}/{path}"


# Singleton instance
_storage_service: Optional[SupabaseStorageService] = None


def get_storage_service() -> SupabaseStorageService:
    """Get or create storage service singleton instance."""
    global _storage_service
    if _storage_service is None:
        _storage_service = SupabaseStorageService()
    return _storage_service
