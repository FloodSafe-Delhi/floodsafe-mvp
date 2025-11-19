export const MAP_CONSTANTS = {
    DARKEST_FLOOD_COLOR: '#519EA2',
    PMTILES_URL: '/tiles.pmtiles',
    BASEMAP_URL: '/basemap.pmtiles',
    CONFIG: {
        center: [77.5777, 12.9776] as [number, number],
        maxZoom: 15,
        hash: true,
        minZoom: 12,
        pitch: 45,
        antialias: true,
        zoom: 12.7,
        maxBounds: [
            [77.199861111, 12.600138889],
            [77.899861111, 13.400138889]
        ] as [[number, number], [number, number]]
    }
} as const;
