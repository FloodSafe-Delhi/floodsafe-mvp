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
