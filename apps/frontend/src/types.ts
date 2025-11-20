export type AlertLevel = 'safe' | 'watch' | 'advisory' | 'warning' | 'emergency';
export type AlertColor = 'green' | 'yellow' | 'orange' | 'red' | 'black';

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

// User type - used across the application
export interface User {
    id: string;
    username: string;
    email: string;
    phone?: string;
    profile_photo_url?: string;
    role: string;
    created_at: string;
    points: number;
    level: number;
    reports_count: number;
    verified_reports_count: number;
    badges?: string[];
    // Profile-specific optional fields
    language?: string;
    notification_push?: boolean;
    notification_sms?: boolean;
    notification_whatsapp?: boolean;
    notification_email?: boolean;
    alert_preferences?: {
        watch: boolean;
        advisory: boolean;
        warning: boolean;
        emergency: boolean;
    };
}

// Location-related types for MapPicker
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
