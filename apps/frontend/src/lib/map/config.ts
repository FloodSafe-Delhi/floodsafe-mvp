import { getCityConfig, type CityKey } from './cityConfigs';

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
