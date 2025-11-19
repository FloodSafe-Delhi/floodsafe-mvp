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

        // Register protocol with MapLibre
        maplibregl.addProtocol('pmtiles', protocol.tile);

        // Use the comprehensive OpenMapTiles style with flood data overlay
        const style = {
            ...mapStyle,
            sources: {
                ...mapStyle.sources,
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
                // Overlay flood data on top of basemap
                {
                    id: 'flood-layer',
                    type: 'fill',
                    source: 'flood-tiles',
                    'source-layer': 'stream_influence_water_difference',
                    paint: {
                        'fill-color': MAP_CONSTANTS.DARKEST_FLOOD_COLOR,
                        'fill-opacity': 0.5
                    }
                }
            ]
        };

        const map = new maplibregl.Map({
            container: containerRef.current,
            style: style as any,
            ...getMapConfig(cityKey)
        });

        map.on('load', () => {
            console.log('✅ Map loaded successfully');
            console.log('📋 Available sources:', Object.keys(map.getStyle().sources));
            console.log('📋 Available layers:', map.getStyle().layers.map(l => l.id));

            // Debug: Check if railway layers exist
            const railwayLayers = map.getStyle().layers.filter(l => l.id.includes('railway'));
            console.log('🚇 Railway layers found:', railwayLayers.map(l => l.id));

            // Debug: Try to query features from transportation layer at current view
            setTimeout(() => {
                const features = map.querySourceFeatures('openmaptiles', {
                    sourceLayer: 'transportation'
                });
                console.log('🚗 Transportation features sample:', features.slice(0, 5));
                const transitFeatures = features.filter(f => f.properties?.class === 'transit');
                console.log('🚇 Transit features:', transitFeatures.length, transitFeatures.slice(0, 3));
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
            maplibregl.removeProtocol('pmtiles');
        };
    }, [cityKey]); // Re-initialize map when city changes

    return { map: mapRef.current, isLoaded };
}
