import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { fetchJson, uploadFile } from './client';
import { User, RouteRequest, RouteResponse, GeocodingResult } from '../../types';
import { validateUsers, validateSensors, validateReports, validateUser } from './validators';

// Types
export interface Sensor {
    id: string;
    location_lat: number; // Note: Backend returns latitude/longitude in response? Let's check DTO.
    // Wait, SensorResponse has latitude/longitude.
    latitude: number;
    longitude: number;
    status: string;
    last_ping?: string;
}

export interface Report {
    id: string;
    description: string;
    latitude: number;
    longitude: number;
    media_url?: string;
    verified: boolean;
    verification_score: number;
    upvotes: number;
    timestamp: string;
    // OTP/Phone verification fields
    phone_verified?: boolean;
    water_depth?: string;
    vehicle_passability?: string;
    iot_validation_score?: number;
    // Gamification fields
    downvotes?: number;
    quality_score?: number;
    verified_at?: string;
}

export interface ReportCreate {
    user_id: string;
    description: string;
    latitude: number;
    longitude: number;
    image?: File;
}

// Re-export User from types for backwards compatibility
export type { User };

// Hooks

export function useSensors() {
    return useQuery({
        queryKey: ['sensors'],
        queryFn: async () => {
            const data = await fetchJson<unknown>('/sensors/');
            return validateSensors(data);
        },
        refetchInterval: 30000, // Default 30 second refresh
    });
}

export function useReports() {
    return useQuery({
        queryKey: ['reports'],
        queryFn: async () => {
            const data = await fetchJson<unknown>('/reports/');
            return validateReports(data);
        },
        refetchInterval: 30000, // Default 30 second refresh
    });
}

export function useUsers() {
    return useQuery({
        queryKey: ['users'],
        queryFn: async () => {
            const data = await fetchJson<unknown>('/users/');
            return validateUsers(data);
        },
    });
}

export interface ActiveReportersStats {
    count: number;
    period_days: number;
}

export interface NearbyReportersStats {
    count: number;
    radius_km: number;
    center: {
        latitude: number;
        longitude: number;
    };
}

export interface LocationDetails {
    location: {
        latitude: number;
        longitude: number;
        radius_meters: number;
    };
    total_reports: number;
    reports: Array<{
        id: string;
        description: string;
        latitude: number;
        longitude: number;
        verified: boolean;
        upvotes: number;
        timestamp: string;
        user_id: string;
    }>;
    reporters: Array<{
        id: string;
        username: string;
        reports_count: number;
        verified_reports_count: number;
        level: number;
    }>;
}

export function useActiveReporters() {
    return useQuery({
        queryKey: ['users', 'stats', 'active-reporters'],
        queryFn: () => fetchJson<ActiveReportersStats>('/users/stats/active-reporters'),
        refetchInterval: 600000, // Refresh every 10 minutes
    });
}

export function useNearbyReporters(latitude: number, longitude: number, radiusKm: number = 5.0) {
    return useQuery({
        queryKey: ['users', 'stats', 'nearby-reporters', latitude, longitude, radiusKm],
        queryFn: () => fetchJson<NearbyReportersStats>(
            `/users/stats/nearby-reporters?latitude=${latitude}&longitude=${longitude}&radius_km=${radiusKm}`
        ),
        refetchInterval: 600000, // Refresh every 10 minutes
        enabled: !!(latitude && longitude), // Only run if coordinates are provided
    });
}

export function useLocationDetails(latitude: number | null, longitude: number | null, radiusMeters: number = 500) {
    return useQuery({
        queryKey: ['reports', 'location', 'details', latitude, longitude, radiusMeters],
        queryFn: () => fetchJson<LocationDetails>(
            `/reports/location/details?latitude=${latitude}&longitude=${longitude}&radius_meters=${radiusMeters}`
        ),
        enabled: !!(latitude && longitude), // Only run if coordinates are provided
    });
}

export function useReportMutation() {
    const queryClient = useQueryClient();

    return useMutation({
        mutationFn: async (data: ReportCreate) => {
            const formData = new FormData();
            formData.append('user_id', data.user_id);
            formData.append('description', data.description);
            formData.append('latitude', data.latitude.toString());
            formData.append('longitude', data.longitude.toString());
            if (data.image) {
                formData.append('image', data.image);
            }
            return uploadFile('/reports/', formData);
        },
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['reports'] });
        },
    });
}

// ============================================================================
// ROUTING HOOKS (Safe route navigation)
// ============================================================================

export function useRouteCalculation() {
    return useMutation({
        mutationFn: async (request: RouteRequest): Promise<RouteResponse> => {
            const response = await fetchJson<RouteResponse>('/routes/calculate', {
                method: 'POST',
                body: JSON.stringify(request),
            });
            return response;
        },
        retry: 1,
    });
}

export function useGeocode(query: string, enabled: boolean = true) {
    return useQuery({
        queryKey: ['geocode', query],
        queryFn: async (): Promise<GeocodingResult[]> => {
            if (!query || query.length < 3) {
                return [];
            }

            const response = await fetch(
                `https://nominatim.openstreetmap.org/search?` +
                `q=${encodeURIComponent(query)}&` +
                `format=json&` +
                `limit=5&` +
                `countrycodes=in&` +
                `addressdetails=1`,
                {
                    headers: {
                        'User-Agent': 'FloodSafe-MVP/1.0'
                    }
                }
            );

            if (!response.ok) {
                throw new Error('Geocoding failed');
            }

            return response.json();
        },
        enabled: enabled && query.length >= 3,
        staleTime: 5 * 60 * 1000, // 5 minutes
        gcTime: 10 * 60 * 1000, // 10 minutes
    });
}
