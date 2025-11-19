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

export interface ReportCreate {
    user_id: string;
    description: string;
    latitude: number;
    longitude: number;
    image?: File;
}

// Hooks

export function useSensors() {
    return useQuery({
        queryKey: ['sensors'],
        queryFn: () => fetchJson<Sensor[]>('/sensors/'),
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
