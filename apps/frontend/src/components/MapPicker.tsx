import React, { useEffect, useRef, useState, useCallback } from 'react';
import maplibregl from 'maplibre-gl';
import { useMap } from '../lib/map/useMap';
import { getCurrentCityConfig, isWithinCityBounds } from '../lib/map/config';
import type { MapPickerProps, LocationWithAddress, GeocodingResult } from '../types';
import { Sheet, SheetContent, SheetHeader, SheetTitle, SheetDescription } from './ui/sheet';
import { Button } from './ui/button';
import { Skeleton } from './ui/skeleton';
import { Plus, Minus, Navigation, MapPin, Check, X, AlertCircle } from 'lucide-react';
import { toast } from 'sonner';

// Debounce utility for reverse geocoding
function useDebounce<T>(value: T, delay: number): T {
    const [debouncedValue, setDebouncedValue] = useState<T>(value);

    useEffect(() => {
        const handler = setTimeout(() => {
            setDebouncedValue(value);
        }, delay);

        return () => {
            clearTimeout(handler);
        };
    }, [value, delay]);

    return debouncedValue;
}

export default function MapPicker({ isOpen, onClose, initialLocation, onLocationSelect }: MapPickerProps) {
    const mapContainer = useRef<HTMLDivElement>(null);
    const { map, isLoaded } = useMap(mapContainer);
    const markerRef = useRef<maplibregl.Marker | null>(null);

    const cityConfig = getCurrentCityConfig();

    // Location state
    const [selectedCoords, setSelectedCoords] = useState<[number, number] | null>(
        initialLocation ? [initialLocation.longitude, initialLocation.latitude] : null
    );

    // Geocoding state
    const [locationName, setLocationName] = useState<string>('');
    const [isGeocoding, setIsGeocoding] = useState(false);
    const [geocodingError, setGeocodingError] = useState<string | null>(null);

    // Map loading timeout
    const [mapLoadError, setMapLoadError] = useState(false);

    // Debounce coordinates to avoid excessive API calls
    const debouncedCoords = useDebounce(selectedCoords, 500);

    // Map load timeout - show error if map doesn't load within 30 seconds
    useEffect(() => {
        if (isLoaded || !isOpen) return;

        const timeout = setTimeout(() => {
            if (!isLoaded) {
                setMapLoadError(true);
                toast.error('Map is taking too long to load. Please check your internet connection.');
            }
        }, 30000); // 30 seconds

        return () => clearTimeout(timeout);
    }, [isLoaded, isOpen]);

    // Initialize map with initial location or city center
    useEffect(() => {
        if (!map || !isLoaded) return;

        const targetCoords = initialLocation
            ? [initialLocation.longitude, initialLocation.latitude] as [number, number]
            : cityConfig.center;

        map.flyTo({
            center: targetCoords,
            zoom: 14,
            duration: 1000
        });

        // Create initial marker
        if (!markerRef.current) {
            const marker = new maplibregl.Marker({
                color: '#ef4444', // Red marker
                draggable: true
            })
                .setLngLat(targetCoords)
                .addTo(map);

            // Handle marker drag
            marker.on('dragend', () => {
                const lngLat = marker.getLngLat();
                handleLocationUpdate(lngLat.lng, lngLat.lat);
            });

            markerRef.current = marker;
            setSelectedCoords(targetCoords);
        }
    }, [map, isLoaded, initialLocation]);

    // Handle map click to move marker
    useEffect(() => {
        if (!map || !isLoaded) return;

        const handleMapClick = (e: maplibregl.MapMouseEvent) => {
            const { lng, lat } = e.lngLat;
            handleLocationUpdate(lng, lat);

            // Move marker to clicked position
            if (markerRef.current) {
                markerRef.current.setLngLat([lng, lat]);
            }
        };

        map.on('click', handleMapClick);

        return () => {
            map.off('click', handleMapClick);
        };
    }, [map, isLoaded]);

    // Handle location update with bounds validation
    const handleLocationUpdate = (longitude: number, latitude: number) => {
        // Validate bounds
        if (!isWithinCityBounds(latitude, longitude)) {
            toast.error(`Location is outside ${cityConfig.name} bounds`);
            setGeocodingError(`Location must be within ${cityConfig.name}`);
            return;
        }

        setSelectedCoords([longitude, latitude]);
        setGeocodingError(null);
    };

    // Reverse geocoding with debounced coordinates
    useEffect(() => {
        if (!debouncedCoords) return;

        const [lng, lat] = debouncedCoords;

        const reverseGeocode = async () => {
            setIsGeocoding(true);
            setGeocodingError(null);

            try {
                const response = await fetch(
                    `https://nominatim.openstreetmap.org/reverse?format=json&lat=${lat}&lon=${lng}&zoom=18&addressdetails=1`,
                    {
                        headers: {
                            'Accept-Language': 'en'
                        }
                    }
                );

                if (!response.ok) {
                    throw new Error('Geocoding request failed');
                }

                const data: GeocodingResult = await response.json();

                // Format location name from address components
                const parts = [];
                if (data.address?.road) parts.push(data.address.road);
                if (data.address?.suburb) parts.push(data.address.suburb);
                if (data.address?.city) parts.push(data.address.city);

                const formattedName = parts.length > 0
                    ? parts.join(', ')
                    : data.display_name;

                setLocationName(formattedName);
            } catch (error) {
                console.error('Reverse geocoding error:', error);
                setGeocodingError('Unable to fetch location name');
                setLocationName(`${lat.toFixed(6)}, ${lng.toFixed(6)}`);
            } finally {
                setIsGeocoding(false);
            }
        };

        reverseGeocode();
    }, [debouncedCoords]);

    // Zoom controls
    const handleZoomIn = useCallback(() => {
        if (map) map.zoomIn();
    }, [map]);

    const handleZoomOut = useCallback(() => {
        if (map) map.zoomOut();
    }, [map]);

    // My Location button
    const handleMyLocation = useCallback(() => {
        if (!map) return;

        if ('geolocation' in navigator) {
            navigator.geolocation.getCurrentPosition(
                (position) => {
                    const { longitude, latitude } = position.coords;

                    // Validate bounds
                    if (!isWithinCityBounds(latitude, longitude)) {
                        toast.error(`Your location is outside ${cityConfig.name} bounds`);
                        return;
                    }

                    map.flyTo({
                        center: [longitude, latitude],
                        zoom: 16,
                        duration: 2000
                    });

                    // Update marker position
                    if (markerRef.current) {
                        markerRef.current.setLngLat([longitude, latitude]);
                    }

                    handleLocationUpdate(longitude, latitude);
                },
                (error) => {
                    console.error('Geolocation error:', error);
                    toast.error('Unable to get your location. Please enable location permissions.');
                },
                {
                    enableHighAccuracy: true,
                    timeout: 10000,
                    maximumAge: 0
                }
            );
        } else {
            toast.error('Geolocation is not supported by your browser.');
        }
    }, [map, cityConfig]);

    // Handle confirm
    const handleConfirm = () => {
        if (!selectedCoords) {
            toast.error('Please select a location on the map');
            return;
        }

        if (geocodingError) {
            toast.error('Please select a valid location');
            return;
        }

        const [longitude, latitude] = selectedCoords;

        const locationData: LocationWithAddress = {
            latitude,
            longitude,
            accuracy: 10, // Map-selected locations have arbitrary good accuracy
            locationName: locationName || `${latitude.toFixed(6)}, ${longitude.toFixed(6)}`
        };

        onLocationSelect(locationData);
        onClose();
        toast.success('Location selected successfully');
    };

    // Handle cancel
    const handleCancel = () => {
        onClose();
    };

    // Cleanup marker on unmount
    useEffect(() => {
        return () => {
            if (markerRef.current) {
                markerRef.current.remove();
                markerRef.current = null;
            }
        };
    }, []);

    return (
        <Sheet open={isOpen} onOpenChange={(open) => !open && handleCancel()}>
            <SheetContent
                side="bottom"
                className="h-[90vh] p-0 flex flex-col"
            >
                <SheetHeader className="px-4 pt-4 pb-2">
                    <SheetTitle className="flex items-center gap-2">
                        <MapPin className="h-5 w-5 text-red-500" />
                        Select Location on Map
                    </SheetTitle>
                    <SheetDescription>
                        Click or drag the marker to select a location in {cityConfig.name}
                    </SheetDescription>
                </SheetHeader>

                {/* Map Container */}
                <div className="flex-1 relative">
                    <div
                        ref={mapContainer}
                        className="w-full h-full"
                    />

                    {/* Map Loading State */}
                    {!isLoaded && !mapLoadError && (
                        <div className="absolute inset-0 bg-gray-100 flex flex-col items-center justify-center z-40">
                            <Skeleton className="w-full h-full" />
                            <div className="absolute flex flex-col items-center gap-2">
                                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
                                <p className="text-sm text-gray-600">Loading flood atlas map...</p>
                            </div>
                        </div>
                    )}

                    {/* Map Load Error State */}
                    {mapLoadError && (
                        <div className="absolute inset-0 bg-gray-50 flex flex-col items-center justify-center z-40 p-6">
                            <AlertCircle className="h-16 w-16 text-red-500 mb-4" />
                            <p className="text-lg font-semibold text-gray-900 mb-2">Failed to Load Map</p>
                            <p className="text-sm text-gray-600 text-center mb-4">
                                The map couldn't load. Please check your internet connection and try again.
                            </p>
                            <Button
                                onClick={() => {
                                    setMapLoadError(false);
                                    window.location.reload();
                                }}
                                variant="outline"
                            >
                                Retry
                            </Button>
                        </div>
                    )}

                    {/* Map Controls */}
                    {isLoaded && (
                        <div className="absolute bottom-4 right-4 flex flex-col gap-2 z-50">
                            <Button
                                size="icon"
                                onClick={handleZoomIn}
                                className="!bg-white !hover:bg-gray-100 !text-gray-800 shadow-xl rounded-full w-11 h-11 border-2 border-gray-300"
                                title="Zoom in"
                            >
                                <Plus className="h-5 w-5" />
                            </Button>
                            <Button
                                size="icon"
                                onClick={handleZoomOut}
                                className="!bg-white !hover:bg-gray-100 !text-gray-800 shadow-xl rounded-full w-11 h-11 border-2 border-gray-300"
                                title="Zoom out"
                            >
                                <Minus className="h-5 w-5" />
                            </Button>
                            <Button
                                size="icon"
                                onClick={handleMyLocation}
                                className="!bg-blue-500 !hover:bg-blue-600 !text-white shadow-xl rounded-full w-11 h-11"
                                title="My location"
                            >
                                <Navigation className="h-5 w-5" />
                            </Button>
                        </div>
                    )}
                </div>

                {/* Location Info & Actions */}
                <div className="px-4 py-4 border-t bg-white space-y-4">
                    {/* Location Name Display */}
                    <div className="space-y-2">
                        <p className="text-sm font-medium text-gray-700">Selected Location:</p>
                        {isGeocoding ? (
                            <Skeleton className="h-6 w-full" />
                        ) : (
                            <p className="text-base font-semibold text-gray-900">
                                {locationName || 'Click on the map to select a location'}
                            </p>
                        )}
                        {selectedCoords && (
                            <p className="text-xs text-gray-500">
                                {selectedCoords[1].toFixed(6)}, {selectedCoords[0].toFixed(6)}
                            </p>
                        )}
                        {geocodingError && (
                            <p className="text-xs text-red-600">{geocodingError}</p>
                        )}
                    </div>

                    {/* Action Buttons */}
                    <div className="flex gap-3">
                        <Button
                            onClick={handleCancel}
                            variant="outline"
                            className="flex-1"
                        >
                            <X className="h-4 w-4 mr-2" />
                            Cancel
                        </Button>
                        <Button
                            onClick={handleConfirm}
                            disabled={!selectedCoords || !!geocodingError || isGeocoding}
                            className="flex-1 bg-blue-600 hover:bg-blue-700"
                        >
                            <Check className="h-4 w-4 mr-2" />
                            Confirm Location
                        </Button>
                    </div>
                </div>
            </SheetContent>
        </Sheet>
    );
}
