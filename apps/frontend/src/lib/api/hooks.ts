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
}

export interface ReportCreate {
    user_id: string;
    description: string;
    latitude: number;
    longitude: number;
    image?: File;
}

export interface User {
    id: string;
    username: string;
    email: string;
    role: string;
    points: number;
    level: number;
    reports_count: number;
    verified_reports_count: number;
    badges: string[];
}

// Hooks

export function useSensors() {
    return useQuery({
        queryKey: ['sensors'],
        queryFn: () => fetchJson<Sensor[]>('/sensors/'),
        refetchInterval: 30000, // Default 30 second refresh
    });
}

export function useReports() {
    return useQuery({
        queryKey: ['reports'],
        queryFn: () => fetchJson<Report[]>('/reports/'),
        refetchInterval: 30000, // Default 30 second refresh
    });
}

export function useUsers() {
    return useQuery({
        queryKey: ['users'],
        queryFn: () => fetchJson<User[]>('/users/'),
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
