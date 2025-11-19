import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { fetchJson, uploadFile } from './client';

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
    phone_verified: boolean;
    water_depth?: string;
    vehicle_passability?: string;
    iot_validation_score: number;
}

export interface ReportCreate {
    user_id: string;
    description: string;
    latitude: number;
    longitude: number;
    phone_number: string;
    phone_verification_token: string;
    water_depth?: string;
    vehicle_passability?: string;
    image: File; // MANDATORY
}

export interface SendOTPRequest {
    phone_number: string;
}

export interface SendOTPResponse {
    success: boolean;
    message: string;
    expires_in: number;
}

export interface VerifyOTPRequest {
    phone_number: string;
    otp_code: string;
}

export interface VerifyOTPResponse {
    verified: boolean;
    message: string;
    token?: string;
}

export interface HyperlocalStatus {
    reports: Report[];
    status: 'safe' | 'caution' | 'warning' | 'critical' | 'unknown';
    area_summary: {
        total_reports: number;
        verified_reports: number;
        avg_water_depth?: string;
        avg_validation_score: number;
        last_updated?: string;
    };
    sensor_summary: {
        sensor_count: number;
        avg_water_level: number;
        max_water_level: number;
        active_sensors: number;
        status: string;
    };
}

// Hooks

export function useSensors() {
    return useQuery({
        queryKey: ['sensors'],
        queryFn: () => fetchJson<Sensor[]>('/sensors/'),
    });
}

export function useReports() {
    return useQuery({
        queryKey: ['reports'],
        queryFn: () => fetchJson<Report[]>('/reports/'),
        refetchInterval: 30000, // Refetch every 30 seconds
    });
}

export function useHyperlocalStatus(lat: number, lng: number, radius: number = 500) {
    return useQuery({
        queryKey: ['hyperlocal', lat, lng, radius],
        queryFn: () => fetchJson<HyperlocalStatus>(`/reports/hyperlocal?lat=${lat}&lng=${lng}&radius=${radius}`),
        enabled: lat !== 0 && lng !== 0, // Only fetch if valid coordinates
    });
}

export function useSendOTP() {
    return useMutation({
        mutationFn: async (data: SendOTPRequest) => {
            return fetchJson<SendOTPResponse>('/otp/send', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data),
            });
        },
    });
}

export function useVerifyOTP() {
    return useMutation({
        mutationFn: async (data: VerifyOTPRequest) => {
            return fetchJson<VerifyOTPResponse>('/otp/verify', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data),
            });
        },
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
            formData.append('phone_number', data.phone_number);
            formData.append('phone_verification_token', data.phone_verification_token);

            if (data.water_depth) {
                formData.append('water_depth', data.water_depth);
            }
            if (data.vehicle_passability) {
                formData.append('vehicle_passability', data.vehicle_passability);
            }

            // Image is mandatory
            formData.append('image', data.image);

            return uploadFile('/reports/', formData);
        },
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['reports'] });
        },
    });
}
