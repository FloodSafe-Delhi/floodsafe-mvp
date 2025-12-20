import { useRef, useState, useCallback } from 'react';
import ExifReader from 'exifreader';
import { Button } from './ui/button';
import { Card } from './ui/card';
import { Badge } from './ui/badge';
import { Camera, ImagePlus, X, MapPin, CheckCircle, AlertTriangle, Loader2 } from 'lucide-react';
import { toast } from 'sonner';
import type { PhotoCaptureProps, PhotoData, PhotoGps } from '../types';

// 100m tolerance in degrees (approximately)
const GPS_TOLERANCE_DEGREES = 0.001;

// Calculate distance between two GPS coordinates
function isWithinTolerance(
    photoGps: PhotoGps,
    reportedLat: number,
    reportedLng: number
): boolean {
    const latDiff = Math.abs(photoGps.lat - reportedLat);
    const lngDiff = Math.abs(photoGps.lng - reportedLng);
    return latDiff <= GPS_TOLERANCE_DEGREES && lngDiff <= GPS_TOLERANCE_DEGREES;
}

// Extract GPS from EXIF data
async function extractExifGps(file: File): Promise<PhotoGps | null> {
    try {
        const arrayBuffer = await file.arrayBuffer();
        const tags = ExifReader.load(arrayBuffer);

        // Check for GPS data
        const latRef = tags['GPSLatitudeRef']?.value;
        const latValue = tags['GPSLatitude']?.description;
        const lngRef = tags['GPSLongitudeRef']?.value;
        const lngValue = tags['GPSLongitude']?.description;

        if (!latValue || !lngValue) {
            return null;
        }

        // Parse latitude
        let lat = parseFloat(latValue);
        if (latRef === 'S' || latRef === 'South') {
            lat = -lat;
        }

        // Parse longitude
        let lng = parseFloat(lngValue);
        if (lngRef === 'W' || lngRef === 'West') {
            lng = -lng;
        }

        if (isNaN(lat) || isNaN(lng)) {
            return null;
        }

        return { lat, lng };
    } catch (error) {
        console.error('Failed to extract EXIF GPS:', error);
        return null;
    }
}

// Get current device GPS
function getCurrentGps(): Promise<PhotoGps> {
    return new Promise((resolve, reject) => {
        if (!('geolocation' in navigator)) {
            reject(new Error('Geolocation not supported'));
            return;
        }

        navigator.geolocation.getCurrentPosition(
            (position) => {
                resolve({
                    lat: position.coords.latitude,
                    lng: position.coords.longitude
                });
            },
            (error) => {
                reject(error);
            },
            {
                enableHighAccuracy: true,
                timeout: 10000,
                maximumAge: 0
            }
        );
    });
}

export default function PhotoCapture({ reportedLocation, onPhotoCapture, photo }: PhotoCaptureProps) {
    const cameraInputRef = useRef<HTMLInputElement>(null);
    const galleryInputRef = useRef<HTMLInputElement>(null);
    const [isProcessing, setIsProcessing] = useState(false);
    const [isGettingGps, setIsGettingGps] = useState(false);

    // Handle camera capture
    const handleCameraCapture = useCallback(async (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (!file) return;

        setIsProcessing(true);
        setIsGettingGps(true);

        try {
            // Get fresh GPS at capture time
            let gps: PhotoGps;
            try {
                gps = await getCurrentGps();
                toast.success('Location captured');
            } catch (gpsError) {
                console.error('GPS error:', gpsError);
                toast.error('Could not get your location. Please enable location services.');
                setIsProcessing(false);
                setIsGettingGps(false);
                // Reset the input
                if (cameraInputRef.current) {
                    cameraInputRef.current.value = '';
                }
                return;
            }

            setIsGettingGps(false);

            // Create preview URL
            const previewUrl = URL.createObjectURL(file);

            // Check if location is verified (within tolerance of reported location)
            const isLocationVerified = reportedLocation
                ? isWithinTolerance(gps, reportedLocation.latitude, reportedLocation.longitude)
                : false;

            if (!isLocationVerified && reportedLocation) {
                toast.warning('Photo location differs from reported location. Report will be flagged for review.');
            }

            const photoData: PhotoData = {
                file,
                gps,
                source: 'camera',
                isLocationVerified,
                previewUrl
            };

            onPhotoCapture(photoData);
        } catch (error) {
            console.error('Camera capture error:', error);
            toast.error('Failed to process photo');
        } finally {
            setIsProcessing(false);
            setIsGettingGps(false);
        }
    }, [reportedLocation, onPhotoCapture]);

    // Handle gallery upload
    const handleGalleryUpload = useCallback(async (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (!file) return;

        setIsProcessing(true);

        try {
            // Extract EXIF GPS from the image
            const gps = await extractExifGps(file);

            if (!gps) {
                toast.error('This photo does not have location data. Please select a geotagged photo or take a new one with your camera.');
                setIsProcessing(false);
                // Reset the input
                if (galleryInputRef.current) {
                    galleryInputRef.current.value = '';
                }
                return;
            }

            // Create preview URL
            const previewUrl = URL.createObjectURL(file);

            // Check if location is verified
            const isLocationVerified = reportedLocation
                ? isWithinTolerance(gps, reportedLocation.latitude, reportedLocation.longitude)
                : false;

            if (!isLocationVerified && reportedLocation) {
                toast.warning('Photo location differs from reported location. Report will be flagged for review.');
            }

            const photoData: PhotoData = {
                file,
                gps,
                source: 'gallery',
                isLocationVerified,
                previewUrl
            };

            onPhotoCapture(photoData);
            toast.success('Photo uploaded successfully');
        } catch (error) {
            console.error('Gallery upload error:', error);
            toast.error('Failed to process photo');
        } finally {
            setIsProcessing(false);
        }
    }, [reportedLocation, onPhotoCapture]);

    // Remove photo
    const handleRemovePhoto = useCallback(() => {
        if (photo?.previewUrl) {
            URL.revokeObjectURL(photo.previewUrl);
        }
        onPhotoCapture(null);
        // Reset inputs
        if (cameraInputRef.current) cameraInputRef.current.value = '';
        if (galleryInputRef.current) galleryInputRef.current.value = '';
    }, [photo, onPhotoCapture]);

    // Trigger camera input
    const triggerCamera = () => {
        cameraInputRef.current?.click();
    };

    // Trigger gallery input
    const triggerGallery = () => {
        galleryInputRef.current?.click();
    };

    return (
        <div className="space-y-4">
            {/* Hidden file inputs - using sr-only to completely hide from view */}
            <input
                ref={cameraInputRef}
                type="file"
                accept="image/*"
                capture="environment"
                onChange={handleCameraCapture}
                className="sr-only"
                id="camera-input"
                name="camera-photo"
                aria-hidden="true"
            />
            <input
                ref={galleryInputRef}
                type="file"
                accept="image/*"
                onChange={handleGalleryUpload}
                className="sr-only"
                id="gallery-input"
                name="gallery-photo"
                aria-hidden="true"
            />

            {/* Photo preview or capture buttons */}
            {photo ? (
                <Card className="relative overflow-hidden">
                    {/* Photo preview */}
                    <div className="relative aspect-[4/3] bg-gray-100">
                        <img
                            src={photo.previewUrl}
                            alt="Captured flood photo"
                            className="w-full h-full object-cover"
                        />

                        {/* Remove button */}
                        <Button
                            variant="destructive"
                            size="icon"
                            className="absolute top-2 right-2 rounded-full w-8 h-8"
                            onClick={handleRemovePhoto}
                        >
                            <X className="h-4 w-4" />
                        </Button>

                        {/* Verification badge */}
                        <div className="absolute bottom-2 left-2">
                            {photo.isLocationVerified ? (
                                <Badge className="bg-green-500 hover:bg-green-600 text-white flex items-center gap-1">
                                    <CheckCircle className="h-3 w-3" />
                                    Location Verified
                                </Badge>
                            ) : (
                                <Badge className="bg-yellow-500 hover:bg-yellow-600 text-white flex items-center gap-1">
                                    <AlertTriangle className="h-3 w-3" />
                                    Location Not Verified
                                </Badge>
                            )}
                        </div>

                        {/* Source badge */}
                        <div className="absolute bottom-2 right-2">
                            <Badge variant="secondary" className="flex items-center gap-1">
                                {photo.source === 'camera' ? (
                                    <Camera className="h-3 w-3" />
                                ) : (
                                    <ImagePlus className="h-3 w-3" />
                                )}
                                {photo.source === 'camera' ? 'Camera' : 'Gallery'}
                            </Badge>
                        </div>
                    </div>

                    {/* GPS coordinates */}
                    <div className="p-3 bg-gray-50 border-t">
                        <div className="flex items-center gap-2 text-sm text-gray-600">
                            <MapPin className="h-4 w-4 text-blue-500" />
                            <span>
                                Photo GPS: {photo.gps.lat.toFixed(6)}, {photo.gps.lng.toFixed(6)}
                            </span>
                        </div>
                        {!photo.isLocationVerified && (
                            <p className="text-xs text-yellow-600 mt-1">
                                Photo location differs from reported location by more than 100m
                            </p>
                        )}
                    </div>
                </Card>
            ) : (
                <div className="border-2 border-dashed border-gray-300 rounded-lg p-6">
                    {isProcessing ? (
                        <div className="text-center py-4">
                            <Loader2 className="w-12 h-12 mx-auto mb-3 text-blue-500 animate-spin" />
                            <p className="text-sm text-gray-600">
                                {isGettingGps ? 'Getting your location...' : 'Processing photo...'}
                            </p>
                        </div>
                    ) : (
                        <div className="text-center">
                            <Camera className="w-12 h-12 mx-auto mb-4 text-gray-400" />
                            <div className="flex flex-col sm:flex-row gap-3 justify-center">
                                <Button
                                    variant="default"
                                    onClick={triggerCamera}
                                    className="flex items-center gap-2"
                                >
                                    <Camera className="h-4 w-4" />
                                    Take Photo
                                </Button>
                                <Button
                                    variant="outline"
                                    onClick={triggerGallery}
                                    className="flex items-center gap-2"
                                >
                                    <ImagePlus className="h-4 w-4" />
                                    Choose from Gallery
                                </Button>
                            </div>
                            <p className="text-xs text-gray-500 mt-3">
                                Gallery photos must have location data (geotagged)
                            </p>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}
