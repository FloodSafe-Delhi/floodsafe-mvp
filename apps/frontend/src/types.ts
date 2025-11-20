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

// ============================================================================
// ROUTING TYPES (Safe route navigation)
// ============================================================================

export interface LocationPoint {
    lng: number;
    lat: number;
}

export type TransportMode = 'driving' | 'walking' | 'metro' | 'combined';
export type RouteType = 'safe' | 'fast' | 'balanced' | 'metro';
export type RiskLevel = 'low' | 'medium' | 'high';

export interface RouteRequest {
    origin: LocationPoint;
    destination: LocationPoint;
    city: 'BLR' | 'DEL';
    mode: TransportMode;
    avoid_risk_levels?: string[];
}

export interface RouteInstruction {
    text: string;
    distance_meters: number;
    duration_seconds?: number;
    maneuver: string;
    location: [number, number]; // [lng, lat]
}

export interface RouteOption {
    id: string;
    type: RouteType;
    city_code: string;
    geometry: GeoJSON.LineString;
    distance_meters: number;
    duration_seconds?: number;
    safety_score: number; // 0-100
    risk_level: RiskLevel;
    flood_intersections: number;
    instructions?: RouteInstruction[];
}

export interface RouteResponse {
    routes: RouteOption[];
    city: string;
    warnings: string[];
}

export interface GeocodingResult {
    display_name: string;
    lat: string;
    lon: string;
    address: {
        road?: string;
        suburb?: string;
        city?: string;
        state?: string;
        country?: string;
    };
}

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
