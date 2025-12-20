/**
 * Geospatial distance and route utilities for FloodSafe navigation
 */

/**
 * Format duration in seconds to human-readable string
 * @param seconds - Duration in seconds
 * @returns Formatted string like "15 min" or "1 hr 20 min"
 */
export function formatDuration(seconds: number): string {
    if (seconds < 60) return '< 1 min';

    const minutes = Math.round(seconds / 60);

    if (minutes < 60) {
        return `${minutes} min`;
    }

    const hours = Math.floor(minutes / 60);
    const remainingMins = minutes % 60;

    if (remainingMins === 0) {
        return `${hours} hr`;
    }

    return `${hours} hr ${remainingMins} min`;
}

/**
 * Format distance in meters to human-readable string
 * @param meters - Distance in meters
 * @returns Formatted string like "5.2 km" or "850 m"
 */
export function formatDistance(meters: number): string {
    if (meters < 1000) {
        return `${Math.round(meters)} m`;
    }
    return `${(meters / 1000).toFixed(1)} km`;
}

/**
 * Calculate distance between two points using Haversine formula
 * @param lat1 - Latitude of first point
 * @param lng1 - Longitude of first point
 * @param lat2 - Latitude of second point
 * @param lng2 - Longitude of second point
 * @returns Distance in meters
 */
export function haversineDistance(
    lat1: number,
    lng1: number,
    lat2: number,
    lng2: number
): number {
    const R = 6371000; // Earth's radius in meters
    const dLat = toRad(lat2 - lat1);
    const dLng = toRad(lng2 - lng1);
    const a =
        Math.sin(dLat / 2) * Math.sin(dLat / 2) +
        Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) *
        Math.sin(dLng / 2) * Math.sin(dLng / 2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
    return R * c;
}

/**
 * Convert degrees to radians
 */
function toRad(deg: number): number {
    return deg * (Math.PI / 180);
}

/**
 * Find hotspots within proximity radius of a location
 * @param userLat - User's latitude
 * @param userLng - User's longitude
 * @param hotspots - Array of hotspot objects with coordinates
 * @param proximityMeters - Proximity radius in meters (default 400m)
 * @returns Array of nearby hotspots with distance, sorted by distance
 */
export function findNearbyHotspots(
    userLat: number,
    userLng: number,
    hotspots: Array<{
        coordinates: [number, number];
        id: number;
        name: string;
        fhi_level: string;
        fhi_color: string;
    }>,
    proximityMeters: number = 400
) {
    return hotspots
        .map(h => ({
            id: h.id,
            name: h.name,
            fhi_level: h.fhi_level,
            fhi_color: h.fhi_color,
            distanceMeters: haversineDistance(
                userLat,
                userLng,
                h.coordinates[1],
                h.coordinates[0]
            )
        }))
        .filter(h => h.distanceMeters <= proximityMeters && h.fhi_level !== 'low')
        .sort((a, b) => a.distanceMeters - b.distanceMeters);
}

/**
 * Check if user is off the planned route
 * @param userLat - User's current latitude
 * @param userLng - User's current longitude
 * @param routeCoords - Array of route coordinates [lng, lat]
 * @param thresholdMeters - Deviation threshold in meters (default 50m)
 * @returns True if user is off route beyond threshold
 */
export function isOffRoute(
    userLat: number,
    userLng: number,
    routeCoords: [number, number][],
    thresholdMeters: number = 50
): boolean {
    if (routeCoords.length === 0) return false;

    let minDistance = Infinity;

    for (const [lng, lat] of routeCoords) {
        const dist = haversineDistance(userLat, userLng, lat, lng);
        if (dist < minDistance) {
            minDistance = dist;
        }
        if (minDistance < thresholdMeters) {
            return false;
        }
    }

    return minDistance > thresholdMeters;
}

/**
 * Find the next turn instruction based on user's current position
 * @param userLat - User's current latitude
 * @param userLng - User's current longitude
 * @param instructions - Array of turn instructions with coordinates
 * @returns Next instruction and distance to it, or null if none
 */
export function findNextInstruction(
    userLat: number,
    userLng: number,
    instructions: Array<{
        coordinates: [number, number];
        instruction: string;
        distance_meters: number;
    }>
): { instruction: any; distanceToNext: number } | null {
    if (instructions.length === 0) return null;

    let closestIdx = 0;
    let minDist = Infinity;

    // Find the instruction closest to user
    for (let i = 0; i < instructions.length; i++) {
        const [lng, lat] = instructions[i].coordinates;
        const dist = haversineDistance(userLat, userLng, lat, lng);
        if (dist < minDist) {
            minDist = dist;
            closestIdx = i;
        }
    }

    // Return next instruction (not the one we're at)
    const nextIdx = minDist < 50 ? closestIdx + 1 : closestIdx;
    if (nextIdx >= instructions.length) {
        return {
            instruction: instructions[instructions.length - 1],
            distanceToNext: 0
        };
    }

    const next = instructions[nextIdx];
    const [lng, lat] = next.coordinates;
    const distanceToNext = haversineDistance(userLat, userLng, lat, lng);

    return { instruction: next, distanceToNext };
}
