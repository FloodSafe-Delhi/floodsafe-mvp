/**
 * Delhi Test Routes for GPS Simulation
 *
 * These routes provide realistic waypoints for testing navigation features
 * without being physically present in Delhi. Waypoints are interpolated
 * to simulate realistic GPS updates every ~50-100 meters.
 */

export interface TestRoute {
    id: string;
    name: string;
    description: string;
    origin: { lat: number; lng: number; name: string };
    destination: { lat: number; lng: number; name: string };
    /** Waypoints in [lng, lat] format (GeoJSON standard) */
    waypoints: [number, number][];
    /** Expected hotspots along the route (for verification) */
    expectedHotspots?: string[];
    /** Estimated duration in seconds */
    estimatedDurationSeconds: number;
    /** Total distance in meters */
    totalDistanceMeters: number;
    /** Suggested speed for simulation (km/h) */
    suggestedSpeedKmh: number;
    /** Point index where deviation test should occur */
    deviationTestIndex?: number;
}

/**
 * Interpolate waypoints between two points to create smooth GPS simulation
 * @param from Starting point [lng, lat]
 * @param to Ending point [lng, lat]
 * @param segmentLengthMeters Desired distance between interpolated points
 * @returns Array of interpolated points
 */
function interpolateWaypoints(
    from: [number, number],
    to: [number, number],
    segmentLengthMeters: number = 50
): [number, number][] {
    const R = 6371000; // Earth radius in meters
    const lat1 = from[1] * Math.PI / 180;
    const lat2 = to[1] * Math.PI / 180;
    const dLat = (to[1] - from[1]) * Math.PI / 180;
    const dLng = (to[0] - from[0]) * Math.PI / 180;

    const a = Math.sin(dLat / 2) ** 2 +
              Math.cos(lat1) * Math.cos(lat2) * Math.sin(dLng / 2) ** 2;
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
    const distance = R * c;

    const numSegments = Math.max(1, Math.ceil(distance / segmentLengthMeters));
    const points: [number, number][] = [];

    for (let i = 0; i <= numSegments; i++) {
        const fraction = i / numSegments;
        const lng = from[0] + (to[0] - from[0]) * fraction;
        const lat = from[1] + (to[1] - from[1]) * fraction;
        points.push([lng, lat]);
    }

    return points;
}

/**
 * Create a route with interpolated waypoints from key points
 */
function createRoute(keyPoints: [number, number][], segmentLength: number = 50): [number, number][] {
    const allPoints: [number, number][] = [];

    for (let i = 0; i < keyPoints.length - 1; i++) {
        const interpolated = interpolateWaypoints(keyPoints[i], keyPoints[i + 1], segmentLength);
        // Avoid duplicating the end point (it becomes the start of next segment)
        if (i < keyPoints.length - 2) {
            interpolated.pop();
        }
        allPoints.push(...interpolated);
    }

    return allPoints;
}

// ============================================================================
// DELHI TEST ROUTES
// ============================================================================

/**
 * Route 1: Connaught Place to India Gate
 * - Short route (~2.5 km)
 * - Passes through central Delhi
 * - May encounter ITO hotspot
 */
const CP_TO_INDIA_GATE_KEY_POINTS: [number, number][] = [
    [77.2167, 28.6315],  // Connaught Place (Rajiv Chowk)
    [77.2195, 28.6298],  // Inner Circle exit
    [77.2215, 28.6275],  // Janpath crossing
    [77.2240, 28.6245],  // Near Janpath Hotel
    [77.2260, 28.6210],  // Windsor Place area
    [77.2280, 28.6175],  // C-Hexagon approach
    [77.2295, 28.6145],  // Near National Stadium
    [77.2295, 28.6129],  // India Gate
];

/**
 * Route 2: Rajiv Chowk Metro to Chandni Chowk
 * - Metro route test (walking segments)
 * - Through Old Delhi
 * - Good for testing metro integration
 */
const RAJIV_CHOWK_TO_CHANDNI_KEY_POINTS: [number, number][] = [
    [77.2195, 28.6328],  // Rajiv Chowk Metro Exit
    [77.2200, 28.6350],  // Connaught Place North
    [77.2210, 28.6400],  // Near Barakhamba Road
    [77.2230, 28.6450],  // Approaching Minto Road
    [77.2260, 28.6490],  // Near New Delhi Railway Station
    [77.2290, 28.6520],  // Approaching Chandni Chowk
    [77.2305, 28.6562],  // Chandni Chowk
];

/**
 * Route 3: Connaught Place to Noida Sector 18
 * - Long route (~15 km)
 * - Crosses Yamuna River
 * - Good for extended testing, deviation tests
 */
const CP_TO_NOIDA_KEY_POINTS: [number, number][] = [
    [77.2167, 28.6315],  // Connaught Place
    [77.2250, 28.6290],  // Mandi House
    [77.2350, 28.6260],  // ITO (HIGH hotspot area)
    [77.2450, 28.6230],  // Near Pragati Maidan
    [77.2550, 28.6200],  // Ring Road
    [77.2650, 28.6150],  // Approaching Yamuna Bank
    [77.2750, 28.6080],  // Akshardham area
    [77.2850, 28.6000],  // Near DND Flyway
    [77.2950, 28.5900],  // Noida Entry
    [77.3100, 28.5800],  // Sector 15
    [77.3200, 28.5750],  // Approaching Sector 18
    [77.3260, 28.5707],  // Noida Sector 18
];

/**
 * Route 4: ITO to Laxmi Nagar (through hotspots)
 * - Specifically designed to pass through flood-prone areas
 * - Good for testing hotspot warnings
 */
const ITO_TO_LAXMI_NAGAR_KEY_POINTS: [number, number][] = [
    [77.2400, 28.6280],  // ITO Junction (HIGH hotspot)
    [77.2450, 28.6260],  // Near IP Estate
    [77.2520, 28.6240],  // Vikas Marg
    [77.2600, 28.6220],  // Approaching Laxmi Nagar
    [77.2680, 28.6200],  // Near Karkardooma
    [77.2750, 28.6180],  // Laxmi Nagar
];

/**
 * Route 5: Short deviation test route
 * - Very short route for quick deviation testing
 * - Deviation at midpoint triggers off-route detection
 */
const DEVIATION_TEST_KEY_POINTS: [number, number][] = [
    [77.2167, 28.6315],  // Start: Connaught Place
    [77.2200, 28.6300],  // Midpoint 1
    [77.2230, 28.6280],  // Midpoint 2 (deviation test here)
    [77.2260, 28.6260],  // Midpoint 3
    [77.2295, 28.6240],  // End: Near Janpath
];

// Build the routes with interpolated waypoints
export const DELHI_TEST_ROUTES: Record<string, TestRoute> = {
    CP_TO_INDIA_GATE: {
        id: 'cp-india-gate',
        name: 'CP to India Gate',
        description: 'Short central Delhi route (~2.5 km, 8 min)',
        origin: { lat: 28.6315, lng: 77.2167, name: 'Connaught Place' },
        destination: { lat: 28.6129, lng: 77.2295, name: 'India Gate' },
        waypoints: createRoute(CP_TO_INDIA_GATE_KEY_POINTS, 50),
        expectedHotspots: ['Janpath'],
        estimatedDurationSeconds: 480, // 8 min
        totalDistanceMeters: 2500,
        suggestedSpeedKmh: 25,
    },

    RAJIV_CHOWK_TO_CHANDNI: {
        id: 'rajiv-chandni',
        name: 'Rajiv Chowk to Chandni Chowk',
        description: 'Metro area route for walking segment tests (~3 km)',
        origin: { lat: 28.6328, lng: 77.2195, name: 'Rajiv Chowk Metro' },
        destination: { lat: 28.6562, lng: 77.2305, name: 'Chandni Chowk' },
        waypoints: createRoute(RAJIV_CHOWK_TO_CHANDNI_KEY_POINTS, 40),
        estimatedDurationSeconds: 600, // 10 min
        totalDistanceMeters: 3000,
        suggestedSpeedKmh: 20, // Walking/auto speed
    },

    CP_TO_NOIDA: {
        id: 'cp-noida',
        name: 'CP to Noida Sector 18',
        description: 'Long route crossing Yamuna (~15 km, 35 min)',
        origin: { lat: 28.6315, lng: 77.2167, name: 'Connaught Place' },
        destination: { lat: 28.5707, lng: 77.3260, name: 'Noida Sector 18' },
        waypoints: createRoute(CP_TO_NOIDA_KEY_POINTS, 100), // Larger segments for long route
        expectedHotspots: ['ITO Junction', 'Yamuna Bank'],
        estimatedDurationSeconds: 2100, // 35 min
        totalDistanceMeters: 15000,
        suggestedSpeedKmh: 35,
        deviationTestIndex: 30, // Test deviation about 1/3 into the route
    },

    ITO_TO_LAXMI_NAGAR: {
        id: 'ito-laxmi',
        name: 'ITO to Laxmi Nagar',
        description: 'Route through flood-prone areas for hotspot testing',
        origin: { lat: 28.6280, lng: 77.2400, name: 'ITO Junction' },
        destination: { lat: 28.6180, lng: 77.2750, name: 'Laxmi Nagar' },
        waypoints: createRoute(ITO_TO_LAXMI_NAGAR_KEY_POINTS, 50),
        expectedHotspots: ['ITO Junction'],
        estimatedDurationSeconds: 600, // 10 min
        totalDistanceMeters: 4000,
        suggestedSpeedKmh: 30,
    },

    DEVIATION_TEST: {
        id: 'deviation-test',
        name: 'Quick Deviation Test',
        description: 'Short route specifically for testing off-route detection',
        origin: { lat: 28.6315, lng: 77.2167, name: 'Connaught Place' },
        destination: { lat: 28.6240, lng: 77.2295, name: 'Near Janpath' },
        waypoints: createRoute(DEVIATION_TEST_KEY_POINTS, 30), // Fine-grained for testing
        estimatedDurationSeconds: 180, // 3 min
        totalDistanceMeters: 1000,
        suggestedSpeedKmh: 20,
        deviationTestIndex: Math.floor(createRoute(DEVIATION_TEST_KEY_POINTS, 30).length / 2),
    },
};

/**
 * Get all available test routes as an array
 */
export function getAllTestRoutes(): TestRoute[] {
    return Object.values(DELHI_TEST_ROUTES);
}

/**
 * Get a test route by ID
 */
export function getTestRouteById(id: string): TestRoute | undefined {
    return Object.values(DELHI_TEST_ROUTES).find(r => r.id === id);
}

/**
 * Delhi landmark coordinates for quick position setting
 */
export const DELHI_LANDMARKS = {
    CONNAUGHT_PLACE: { lat: 28.6315, lng: 77.2167, name: 'Connaught Place' },
    INDIA_GATE: { lat: 28.6129, lng: 77.2295, name: 'India Gate' },
    ITO_JUNCTION: { lat: 28.6280, lng: 77.2400, name: 'ITO Junction (Hotspot)' },
    RAJIV_CHOWK_METRO: { lat: 28.6328, lng: 77.2195, name: 'Rajiv Chowk Metro' },
    CHANDNI_CHOWK: { lat: 28.6562, lng: 77.2305, name: 'Chandni Chowk' },
    NOIDA_SECTOR_18: { lat: 28.5707, lng: 77.3260, name: 'Noida Sector 18' },
    JANPATH: { lat: 28.6262, lng: 77.2187, name: 'Janpath (Deviation Test)' },
    MANDI_HOUSE: { lat: 28.6260, lng: 77.2305, name: 'Mandi House' },
    YAMUNA_BANK: { lat: 28.6080, lng: 77.2750, name: 'Yamuna Bank' },
    LAXMI_NAGAR: { lat: 28.6180, lng: 77.2750, name: 'Laxmi Nagar' },
};

/**
 * Get deviation position for a route
 * Returns a position 100m perpendicular to the route at the deviation test index
 */
export function getDeviationPosition(route: TestRoute, offsetMeters: number = 100): { lat: number; lng: number } | null {
    const idx = route.deviationTestIndex ?? Math.floor(route.waypoints.length / 2);
    if (idx >= route.waypoints.length) return null;

    const point = route.waypoints[idx];
    // Simple perpendicular offset (east)
    const offsetDegrees = offsetMeters / 111320;
    return {
        lat: point[1],
        lng: point[0] + offsetDegrees,
    };
}
