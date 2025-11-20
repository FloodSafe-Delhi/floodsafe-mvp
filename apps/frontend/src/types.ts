export interface FloodAlert {
    id: string;
    level: 'critical' | 'warning' | 'watch' | 'safe';
    location: string;
    description: string;
    timeUntil: string;
    confidence: number;
    isActive: boolean;
    color: 'red' | 'orange' | 'yellow' | 'green';
    coordinates: [number, number];
}

export type WaterDepth = 'ankle' | 'knee' | 'waist' | 'impassable';
export type VehiclePassability = 'all' | 'high-clearance' | 'none';

// Location-related types
export interface LocationCoordinates {
    latitude: number;
    longitude: number;
}

export interface LocationData extends LocationCoordinates {
    accuracy: number;
}

export interface LocationWithAddress extends LocationData {
    locationName: string;
}

// Map Picker types
export interface MapPickerProps {
    isOpen: boolean;
    onClose: () => void;
    initialLocation: LocationData | null;
    onLocationSelect: (location: LocationWithAddress) => void;
}

export interface GeocodingResult {
    display_name: string;
    lat: string;
    lon: string;
    address?: {
        road?: string;
        suburb?: string;
        city?: string;
        state?: string;
        country?: string;
    };
}

export interface MapBounds {
    minLng: number;
    maxLng: number;
    minLat: number;
    maxLat: number;
}

export interface CityConfig {
    name: string;
    center: [number, number];
    bounds: MapBounds;
    defaultZoom: number;
    maxZoom: number;
    minZoom: number;
}
