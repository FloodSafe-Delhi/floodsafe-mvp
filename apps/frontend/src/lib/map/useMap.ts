import { useEffect, useRef, useState } from 'react';
import maplibregl from 'maplibre-gl';
import { Protocol, PMTiles } from 'pmtiles';
import { MAP_CONSTANTS, getMapConfig } from './config';
import { getCityConfig, type CityKey } from './cityConfigs';
import mapStyle from './styles.json';

export function useMap(
    containerRef: React.RefObject<HTMLDivElement>,
    cityKey: CityKey = MAP_CONSTANTS.DEFAULT_CITY
) {
    const mapRef = useRef<maplibregl.Map | null>(null);
    const [isLoaded, setIsLoaded] = useState(false);

    useEffect(() => {
        if (!containerRef.current || mapRef.current) return;

        const cityConfig = getCityConfig(cityKey);

        // Initialize PMTiles protocol
        const protocol = new Protocol();

        // Add basemap PMTiles for selected city
        const basemapPMTiles = new PMTiles(cityConfig.pmtiles.basemap);
        protocol.add(basemapPMTiles);

        // Add flood tiles PMTiles for selected city
        const floodPMTiles = new PMTiles(cityConfig.pmtiles.flood);
        protocol.add(floodPMTiles);

        // Register protocol with MapLibre - only if not already registered
        // This prevents errors in React Strict Mode or multiple map instances
        let protocolAdded = false;
        try {
            maplibregl.addProtocol('pmtiles', protocol.tile);
            protocolAdded = true;
        } catch (error) {
            // Protocol already registered - this is fine, reuse existing
            console.log('PMTiles protocol already registered, reusing existing');
        }

        // Use the comprehensive OpenMapTiles style with flood data overlay
        const style = {
            ...mapStyle,
            sources: {
                ...mapStyle.sources,
                // Override the basemap source with city-specific PMTiles
                'openmaptiles': {
                    type: 'vector',
                    url: `pmtiles://${cityConfig.pmtiles.basemap}`
                },
                // Add flood visualization data for selected city
                'flood-tiles': {
                    type: 'vector',
                    url: `pmtiles://${cityConfig.pmtiles.flood}`,
                    attribution: '© <a href="https://openstreetmap.org">OpenStreetMap</a>'
                },
                // Add metro lines from GeoJSON for selected city
                'metro-lines': {
                    type: 'geojson',
                    data: cityConfig.metro.lines
                },
                // Add metro stations from GeoJSON for selected city
                'metro-stations': {
                    type: 'geojson',
                    data: cityConfig.metro.stations
                }
            },
            layers: [
                ...mapStyle.layers.filter(l => l.id !== 'railway-transit' && l.id !== 'railway'),
                // Add metro lines layer
                {
                    id: 'metro-lines-layer',
                    type: 'line',
                    source: 'metro-lines',
                    layout: {
                        'visibility': 'visible',
                        'line-join': 'round',
                        'line-cap': 'round'
                    },
                    paint: {
                        'line-color': ['get', 'colour'],
                        'line-width': 4,
                        'line-opacity': 1
                    }
                },
                // Add metro stations layer
                {
                    id: 'metro-stations-layer',
                    type: 'circle',
                    source: 'metro-stations',
                    layout: {
                        'visibility': 'visible'
                    },
                    paint: {
                        'circle-radius': 4,
                        'circle-color': '#ffffff',
                        'circle-stroke-width': 2,
                        'circle-stroke-color': ['get', 'color']
                    }
                },
                // Add metro station names layer
                {
                    id: 'metro-station-names-layer',
                    type: 'symbol',
                    source: 'metro-stations',
                    minzoom: 12,
                    layout: {
                        'visibility': 'visible',
                        'text-field': ['get', 'name'],
                        'text-font': ['Open Sans Bold', 'Arial Unicode MS Bold'],
                        'text-size': 12,
                        'text-offset': [0, 1.5],
                        'text-anchor': 'top'
                    },
                    paint: {
                        'text-color': '#333333',
                        'text-halo-color': '#ffffff',
                        'text-halo-width': 2
                    }
                },
                // Overlay flood data on top of basemap - graduated color scheme for flood risk
                {
                    id: 'flood-layer',
                    type: 'fill',
                    source: 'flood-tiles',
                    'source-layer': 'stream_influence_water_difference',
                    paint: {
                        // Use data-driven styling based on VALUE property (1-4 scale from DEM processing)
                        'fill-color': [
                            'interpolate',
                            ['linear'],
                            ['get', 'VALUE'],
                            1, '#FFFFCC',  // Light yellow - lowest flood risk
                            2, '#A1DAB4',  // Light green
                            3, '#41B6C4',  // Teal
                            4, '#225EA8'   // Dark blue - highest flood risk
                        ],
                        'fill-opacity': 0.6
                    }
                }
            ]
        };

        const map = new maplibregl.Map({
            container: containerRef.current,
            style: style as maplibregl.StyleSpecification,
            ...getMapConfig(cityKey)
        });

        map.on('load', () => {
            console.log('✅ Map loaded successfully');

            // Safely access map style with type guards
            const mapStyle = map.getStyle();
            if (mapStyle?.sources) {
                console.log('📋 Available sources:', Object.keys(mapStyle.sources));
            }

            if (mapStyle?.layers && Array.isArray(mapStyle.layers)) {
                console.log('📋 Available layers:', mapStyle.layers.map(l => l.id));

                // Debug: Check if railway layers exist
                const railwayLayers = mapStyle.layers.filter(l =>
                    typeof l.id === 'string' && l.id.includes('railway')
                );
                console.log('🚇 Railway layers found:', railwayLayers.map(l => l.id));
            }

            // Debug: Try to query features from transportation layer at current view
            setTimeout(() => {
                try {
                    const style = map.getStyle();
                    if (!style?.sources || typeof style.sources !== 'object') {
                        console.log('ℹ️ Map style not ready yet');
                        return;
                    }

                    const sourceKeys = Object.keys(style.sources);
                    const sourceKey = sourceKeys.find(key =>
                        typeof key === 'string' && (key.includes('basemap') || key.includes('openmaptiles'))
                    );

                    if (sourceKey) {
                        const source = map.getSource(sourceKey);
                        if (!source) {
                            console.log('ℹ️ Source not found:', sourceKey);
                            return;
                        }

                        const features = map.querySourceFeatures(sourceKey, {
                            sourceLayer: 'transportation'
                        });

                        if (features && Array.isArray(features) && features.length > 0) {
                            console.log('🚗 Transportation features sample:', features.slice(0, 5));
                            const transitFeatures = features.filter(f =>
                                f.properties && typeof f.properties === 'object' && f.properties.class === 'transit'
                            );
                            if (transitFeatures.length > 0) {
                                console.log('🚇 Transit features:', transitFeatures.length, transitFeatures.slice(0, 3));
                            }
                        }
                    } else {
                        console.log('ℹ️ No basemap source found for querying transportation features');
                    }
                } catch (error) {
                    console.log('ℹ️ Could not query transportation features:', error);
                }
            }, 2000);

            setIsLoaded(true);
        });

        map.on('error', (e) => {
            console.error('❌ Map error:', e);
            // Gracefully handle missing files
            if (e.error?.message?.includes('404') || e.error?.message?.includes('Failed to fetch')) {
                console.warn(`⚠️ Some map resources for ${cityConfig.displayName} are not available. Using fallback.`);
            }
        });

        mapRef.current = map;

        return () => {
            map.remove();
            mapRef.current = null;
            // Only remove protocol if we added it
            if (protocolAdded) {
                try {
                    maplibregl.removeProtocol('pmtiles');
                } catch (error) {
                    // Protocol might have been removed already - this is fine
                    console.log('PMTiles protocol already removed or never added');
                }
            }
        };
    }, [cityKey]); // Re-initialize map when city changes

    return { map: mapRef.current, isLoaded };
}
