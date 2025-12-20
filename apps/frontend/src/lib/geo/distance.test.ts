/**
 * Unit tests for geospatial distance and route utilities
 */
import { describe, it, expect } from 'vitest';
import {
    haversineDistance,
    isOffRoute,
    findNearbyHotspots,
    findNextInstruction,
    formatDuration,
    formatDistance,
} from './distance';

// Delhi test coordinates
const CONNAUGHT_PLACE = { lat: 28.6315, lng: 77.2167 };
const INDIA_GATE = { lat: 28.6129, lng: 77.2295 };
const ITO_JUNCTION = { lat: 28.6280, lng: 77.2400 };
const RAJIV_CHOWK = { lat: 28.6328, lng: 77.2195 };

describe('haversineDistance', () => {
    it('calculates correct distance between CP and India Gate (~2.3 km)', () => {
        const dist = haversineDistance(
            CONNAUGHT_PLACE.lat,
            CONNAUGHT_PLACE.lng,
            INDIA_GATE.lat,
            INDIA_GATE.lng
        );
        // Known distance is approximately 2.3 km
        expect(dist).toBeGreaterThan(2200);
        expect(dist).toBeLessThan(2500);
    });

    it('returns 0 for same point', () => {
        const dist = haversineDistance(
            CONNAUGHT_PLACE.lat,
            CONNAUGHT_PLACE.lng,
            CONNAUGHT_PLACE.lat,
            CONNAUGHT_PLACE.lng
        );
        expect(dist).toBe(0);
    });

    it('calculates short distances accurately (<100m)', () => {
        // Two points approximately 50m apart
        const lat1 = 28.6315;
        const lng1 = 77.2167;
        const lat2 = 28.6319; // ~45m north
        const lng2 = 77.2167;

        const dist = haversineDistance(lat1, lng1, lat2, lng2);
        expect(dist).toBeGreaterThan(40);
        expect(dist).toBeLessThan(60);
    });

    it('is symmetric (same distance A→B and B→A)', () => {
        const distAB = haversineDistance(
            CONNAUGHT_PLACE.lat,
            CONNAUGHT_PLACE.lng,
            INDIA_GATE.lat,
            INDIA_GATE.lng
        );
        const distBA = haversineDistance(
            INDIA_GATE.lat,
            INDIA_GATE.lng,
            CONNAUGHT_PLACE.lat,
            CONNAUGHT_PLACE.lng
        );
        expect(distAB).toBeCloseTo(distBA, 5);
    });
});

describe('isOffRoute', () => {
    const route: [number, number][] = [
        [77.2167, 28.6315], // CP [lng, lat]
        [77.2200, 28.6280], // Midpoint
        [77.2295, 28.6129], // India Gate
    ];

    it('returns false when exactly on route', () => {
        expect(isOffRoute(28.6315, 77.2167, route, 50)).toBe(false);
    });

    it('returns false when close to route (<50m)', () => {
        // Slightly off from CP
        expect(isOffRoute(28.6316, 77.2168, route, 50)).toBe(false);
    });

    it('returns true when far from route (>50m)', () => {
        // ITO Junction is not on the CP→India Gate route
        expect(isOffRoute(ITO_JUNCTION.lat, ITO_JUNCTION.lng, route, 50)).toBe(true);
    });

    it('returns false for empty route', () => {
        expect(isOffRoute(28.6315, 77.2167, [], 50)).toBe(false);
    });

    it('respects custom threshold', () => {
        // Point ~30m from route
        const nearPoint = { lat: 28.6317, lng: 77.2170 };
        expect(isOffRoute(nearPoint.lat, nearPoint.lng, route, 20)).toBe(true);
        expect(isOffRoute(nearPoint.lat, nearPoint.lng, route, 100)).toBe(false);
    });
});

describe('findNearbyHotspots', () => {
    const hotspots = [
        {
            id: 1,
            name: 'ITO Junction',
            fhi_level: 'high',
            fhi_color: '#f59e0b',
            coordinates: [77.2400, 28.6280] as [number, number],
        },
        {
            id: 2,
            name: 'Mandi House',
            fhi_level: 'moderate',
            fhi_color: '#eab308',
            coordinates: [77.2305, 28.6260] as [number, number],
        },
        {
            id: 3,
            name: 'Far Away Place',
            fhi_level: 'extreme',
            fhi_color: '#ef4444',
            coordinates: [77.3000, 28.7000] as [number, number],
        },
        {
            id: 4,
            name: 'Low Risk Area',
            fhi_level: 'low',
            fhi_color: '#22c55e',
            coordinates: [77.2200, 28.6300] as [number, number],
        },
    ];

    it('finds hotspots within 500m radius', () => {
        // Position near Mandi House (within 500m)
        const nearby = findNearbyHotspots(28.6260, 77.2320, hotspots, 500);
        expect(nearby.length).toBeGreaterThanOrEqual(1);
        expect(nearby.some(h => h.name === 'Mandi House')).toBe(true);
    });

    it('excludes low FHI hotspots', () => {
        // Position near Low Risk Area
        const nearby = findNearbyHotspots(28.6300, 77.2200, hotspots, 500);
        expect(nearby.some(h => h.name === 'Low Risk Area')).toBe(false);
    });

    it('excludes hotspots outside radius', () => {
        const nearby = findNearbyHotspots(28.6280, 77.2350, hotspots, 400);
        expect(nearby.some(h => h.name === 'Far Away Place')).toBe(false);
    });

    it('sorts results by distance (closest first)', () => {
        const nearby = findNearbyHotspots(28.6280, 77.2350, hotspots, 5000);
        if (nearby.length >= 2) {
            expect(nearby[0].distanceMeters).toBeLessThanOrEqual(nearby[1].distanceMeters);
        }
    });

    it('returns empty array for no matches', () => {
        const nearby = findNearbyHotspots(28.5000, 77.1000, hotspots, 100);
        expect(nearby).toHaveLength(0);
    });

    it('includes distance in result objects', () => {
        const nearby = findNearbyHotspots(28.6260, 77.2305, hotspots, 500);
        expect(nearby.length).toBeGreaterThan(0);
        expect(nearby[0]).toHaveProperty('distanceMeters');
        expect(typeof nearby[0].distanceMeters).toBe('number');
    });
});

describe('findNextInstruction', () => {
    const instructions = [
        {
            coordinates: [77.2167, 28.6315] as [number, number],
            instruction: 'Head south on Janpath',
            distance_meters: 500,
        },
        {
            coordinates: [77.2200, 28.6280] as [number, number],
            instruction: 'Turn right onto Rajpath',
            distance_meters: 800,
        },
        {
            coordinates: [77.2295, 28.6129] as [number, number],
            instruction: 'Arrive at India Gate',
            distance_meters: 0,
        },
    ];

    it('returns next instruction when at a turn point', () => {
        // At CP (first instruction point)
        const result = findNextInstruction(28.6315, 77.2167, instructions);
        expect(result).not.toBeNull();
        expect(result!.instruction.instruction).toBe('Turn right onto Rajpath');
    });

    it('returns closest upcoming instruction', () => {
        // Between first and second instruction
        const result = findNextInstruction(28.6290, 77.2180, instructions);
        expect(result).not.toBeNull();
        expect(result!.distanceToNext).toBeGreaterThan(0);
    });

    it('returns last instruction at destination', () => {
        // At India Gate (last instruction)
        const result = findNextInstruction(28.6129, 77.2295, instructions);
        expect(result).not.toBeNull();
        expect(result!.instruction.instruction).toBe('Arrive at India Gate');
    });

    it('returns null for empty instructions', () => {
        const result = findNextInstruction(28.6315, 77.2167, []);
        expect(result).toBeNull();
    });

    it('includes distance to next instruction', () => {
        const result = findNextInstruction(28.6300, 77.2180, instructions);
        expect(result).not.toBeNull();
        expect(typeof result!.distanceToNext).toBe('number');
        expect(result!.distanceToNext).toBeGreaterThanOrEqual(0);
    });
});

describe('formatDuration', () => {
    it('formats seconds < 60 as "< 1 min"', () => {
        expect(formatDuration(30)).toBe('< 1 min');
        expect(formatDuration(59)).toBe('< 1 min');
    });

    it('formats minutes correctly', () => {
        expect(formatDuration(60)).toBe('1 min');
        expect(formatDuration(720)).toBe('12 min');
        expect(formatDuration(3540)).toBe('59 min');
    });

    it('formats hours correctly', () => {
        expect(formatDuration(3600)).toBe('1 hr');
        expect(formatDuration(7200)).toBe('2 hr');
    });

    it('formats hours and minutes', () => {
        expect(formatDuration(3660)).toBe('1 hr 1 min');
        expect(formatDuration(5400)).toBe('1 hr 30 min');
        expect(formatDuration(8100)).toBe('2 hr 15 min');
    });

    it('handles edge case of 0 seconds', () => {
        expect(formatDuration(0)).toBe('< 1 min');
    });
});

describe('formatDistance', () => {
    it('formats meters < 1000 correctly', () => {
        expect(formatDistance(50)).toBe('50 m');
        expect(formatDistance(500)).toBe('500 m');
        expect(formatDistance(999)).toBe('999 m');
    });

    it('formats kilometers correctly', () => {
        expect(formatDistance(1000)).toBe('1.0 km');
        expect(formatDistance(2300)).toBe('2.3 km');
        expect(formatDistance(5678)).toBe('5.7 km');
    });

    it('rounds meters to nearest integer', () => {
        expect(formatDistance(123.7)).toBe('124 m');
    });

    it('formats kilometers to one decimal place', () => {
        expect(formatDistance(2345)).toBe('2.3 km');
        expect(formatDistance(2355)).toBe('2.4 km');
    });

    it('handles 0 meters', () => {
        expect(formatDistance(0)).toBe('0 m');
    });
});
