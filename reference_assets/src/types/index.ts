export type AlertLevel = 'safe' | 'watch' | 'advisory' | 'warning' | 'emergency';

export type AlertColor = 'green' | 'yellow' | 'orange' | 'red' | 'black';

export interface FloodAlert {
  id: string;
  location: string;
  level: AlertLevel;
  color: AlertColor;
  title: string;
  expectedTime: string;
  confidence: number;
  waterDepth: WaterDepth;
  coordinates: [number, number];
  description: string;
  impact: string;
  source: string;
  isActive: boolean;
  timeUntil?: string;
}

export type WaterDepth = 'ankle' | 'knee' | 'waist' | 'impassable';

export type VehiclePassability = 'all' | 'high-clearance' | 'none';

export interface CommunityReport {
  id: string;
  userId: string;
  userName: string;
  timestamp: string;
  location: string;
  coordinates: [number, number];
  waterDepth: WaterDepth;
  vehiclePassability: VehiclePassability;
  photoUrl?: string;
  verified: boolean;
  accuracy: number;
}

export interface SafeRoute {
  id: string;
  name: string;
  additionalTime: string;
  status: string;
  description: string;
}

export interface UserProfile {
  phone: string;
  role: string;
  joinDate: string;
  score: number;
  badge: string;
  reportsSubmitted: number;
  reportsVerified: number;
  reportsPending: number;
}

export interface WatchArea {
  name: string;
  status: AlertLevel;
  color: AlertColor;
}
