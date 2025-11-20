import { getCityConfig, isWithinCityBounds as checkBounds, type CityKey } from './cityConfigs';

/**
 * Shared constants that apply to all cities
 */
export const MAP_CONSTANTS = {
    DARKEST_FLOOD_COLOR: '#519EA2',
    DEFAULT_CITY: 'bangalore' as CityKey,
} as const;

/**
 * Generate city-specific map configuration
 * @param cityKey - The city to get configuration for
 * @returns MapLibre GL configuration object
 */
export function getMapConfig(cityKey: CityKey = MAP_CONSTANTS.DEFAULT_CITY) {
    const city = getCityConfig(cityKey);

    return {
        center: city.center,
        maxZoom: city.maxZoom,
        hash: true,
        minZoom: city.minZoom,
        pitch: city.pitch || 45,
        antialias: true,
        zoom: city.zoom,
        maxBounds: city.bounds
    };
}

/**
 * Get current city configuration (defaults to DEFAULT_CITY)
 * @returns The configuration for the current active city
 */
export function getCurrentCityConfig() {
    return getCityConfig(MAP_CONSTANTS.DEFAULT_CITY);
}

/**
 * Check if coordinates are within city bounds
 * Re-export from cityConfigs for backward compatibility
 */
export function isWithinCityBounds(
    latitude: number,
    longitude: number,
    cityKey: CityKey = MAP_CONSTANTS.DEFAULT_CITY
): boolean {
    return checkBounds(longitude, latitude, cityKey);
}
