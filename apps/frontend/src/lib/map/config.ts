import type { CityConfig } from '../../types';

// City configurations - scalable for multiple cities
export const CITY_CONFIGS: Record<string, CityConfig> = {
    bangalore: {
        name: 'Bangalore',
        center: [77.5777, 12.9776],
        bounds: {
            minLng: 77.199861111,
            maxLng: 77.899861111,
            minLat: 12.600138889,
            maxLat: 13.400138889
        },
        defaultZoom: 12.7,
        maxZoom: 15,
        minZoom: 12
    },
    delhi: {
        name: 'Delhi',
        center: [77.2090, 28.6139],
        bounds: {
            minLng: 76.8388,
            maxLng: 77.5493,
            minLat: 28.4040,
            maxLat: 28.8833
        },
        defaultZoom: 11.5,
        maxZoom: 15,
        minZoom: 10
    }
} as const;

// Current active city - can be made dynamic via environment variable or user selection
export const ACTIVE_CITY: keyof typeof CITY_CONFIGS = 'bangalore';

// Get active city configuration
export const getCurrentCityConfig = (): CityConfig => {
    return CITY_CONFIGS[ACTIVE_CITY];
};

// Check if coordinates are within city bounds
export const isWithinCityBounds = (
    latitude: number,
    longitude: number,
    cityKey: keyof typeof CITY_CONFIGS = ACTIVE_CITY
): boolean => {
    const city = CITY_CONFIGS[cityKey];
    return (
        longitude >= city.bounds.minLng &&
        longitude <= city.bounds.maxLng &&
        latitude >= city.bounds.minLat &&
        latitude <= city.bounds.maxLat
    );
};

export const MAP_CONSTANTS = {
    DARKEST_FLOOD_COLOR: '#519EA2',
    PMTILES_URL: '/tiles.pmtiles',
    BASEMAP_URL: '/basemap.pmtiles',
    CONFIG: {
        center: getCurrentCityConfig().center as [number, number],
        maxZoom: getCurrentCityConfig().maxZoom,
        hash: true,
        minZoom: getCurrentCityConfig().minZoom,
        pitch: 45,
        antialias: true,
        zoom: getCurrentCityConfig().defaultZoom,
        maxBounds: [
            [getCurrentCityConfig().bounds.minLng, getCurrentCityConfig().bounds.minLat],
            [getCurrentCityConfig().bounds.maxLng, getCurrentCityConfig().bounds.maxLat]
        ] as [[number, number], [number, number]]
    }
} as const;
