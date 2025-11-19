import { FloodAlert } from '../types';

export const mockAlerts: FloodAlert[] = [
    {
        id: '1',
        level: 'critical',
        location: 'Yamuna Bank',
        description: 'Water levels rising rapidly due to heavy rainfall upstream.',
        timeUntil: '2h 15m',
        confidence: 85,
        isActive: true,
        color: 'red',
        coordinates: [77.28, 28.61]
    },
    {
        id: '2',
        level: 'warning',
        location: 'ITO Bridge',
        description: 'Moderate water logging reported.',
        timeUntil: '4h',
        confidence: 60,
        isActive: true,
        color: 'orange',
        coordinates: [77.24, 28.62]
    }
];
