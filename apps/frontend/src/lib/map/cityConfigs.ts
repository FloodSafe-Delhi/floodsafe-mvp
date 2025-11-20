export interface CityConfig {
    code: 'BLR' | 'DEL';
    name: string;
    displayName: string;
    center: [number, number];
    zoom: number;
    minZoom: number;
    maxZoom: number;
    pitch: number;
    maxBounds: [[number, number], [number, number]];
    pmtiles: {
        basemap: string;
        flood: string;
    };
    metro: {
        enabled: boolean;
        lines: string;
        stations: string;
        lineCount: number;
        stationCount: number;
    };
}

export const CITY_CONFIGS: Record<'BLR' | 'DEL', CityConfig> = {
    BLR: {
        code: 'BLR',
        name: 'Bangalore',
        displayName: 'Bengaluru, Karnataka',
        center: [77.5777, 12.9776],
        zoom: 12.7,
        minZoom: 12,
        maxZoom: 15,
        pitch: 45,
        maxBounds: [
            [77.199861111, 12.600138889],
            [77.899861111, 13.400138889]
        ],
        pmtiles: {
            basemap: '/basemap.pmtiles',
            flood: '/tiles.pmtiles'
        },
        metro: {
            enabled: false, // Will be enabled when Namma Metro data is added
            lines: '/bangalore-metro-lines.geojson',
            stations: '/bangalore-metro-stations.geojson',
            lineCount: 2, // Purple, Green
            stationCount: 42
        }
    },
    DEL: {
        code: 'DEL',
        name: 'Delhi',
        displayName: 'New Delhi, NCR',
        center: [77.2090, 28.6139],  // India Gate / Central Delhi
        zoom: 12,
        minZoom: 11,
        maxZoom: 15,
        pitch: 45,
        maxBounds: [
            [76.84, 28.40],   // Southwest corner (extended for metro coverage)
            [77.50, 28.88]    // Northeast corner
        ],
        pmtiles: {
            basemap: '/delhi-basemap.pmtiles',
            flood: '/delhi-tiles.pmtiles'
        },
        metro: {
            enabled: true,
            lines: '/delhi-metro-lines.geojson',
            stations: '/delhi-metro-stations.geojson',
            lineCount: 8, // Red, Yellow, Blue, Green, Violet, Orange, Magenta, Pink
            stationCount: 15 // Key stations loaded
        }
    }
};

/**
 * Auto-detect city from user coordinates
 */
export function detectCityFromCoords(lng: number, lat: number): 'BLR' | 'DEL' | null {
    for (const city of Object.values(CITY_CONFIGS)) {
        const [[minLng, minLat], [maxLng, maxLat]] = city.maxBounds;
        if (lng >= minLng && lng <= maxLng && lat >= minLat && lat <= maxLat) {
            return city.code;
        }
    }
    return null;
}

/**
 * Get city from localStorage or default to Bangalore
 */
export function getCurrentCity(): CityConfig {
    if (typeof window === 'undefined') {
        return CITY_CONFIGS.BLR; // SSR fallback
    }

    const savedCity = localStorage.getItem('floodsafe_city');
    if (savedCity === 'BLR' || savedCity === 'DEL') {
        return CITY_CONFIGS[savedCity];
    }

    return CITY_CONFIGS.BLR; // Default to Bangalore
}

/**
 * Save selected city to localStorage
 */
export function setCurrentCity(cityCode: 'BLR' | 'DEL'): void {
    if (typeof window === 'undefined') return;
    localStorage.setItem('floodsafe_city', cityCode);
}

/**
 * Get all available cities
 */
export function getAvailableCities(): CityConfig[] {
    return Object.values(CITY_CONFIGS);
}
