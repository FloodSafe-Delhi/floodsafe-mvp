import { useEffect, useRef, useState } from 'react';
import maplibregl from 'maplibre-gl';
import { Protocol, PMTiles } from 'pmtiles';
import { MAP_CONSTANTS } from './config';
import mapStyle from './styles.json';

export function useMap(containerRef: React.RefObject<HTMLDivElement>) {
    const mapRef = useRef<maplibregl.Map | null>(null);
    const [isLoaded, setIsLoaded] = useState(false);

    useEffect(() => {
        if (!containerRef.current || mapRef.current) return;

        const protocol = new Protocol();
        maplibregl.addProtocol('pmtiles', protocol.tile);

        // Use the comprehensive OpenMapTiles style with flood data overlay
        const style = {
            ...mapStyle,
            sources: {
                ...mapStyle.sources,
                // Add flood visualization data
                'flood-tiles': {
                    type: 'vector',
                    url: `pmtiles://${MAP_CONSTANTS.PMTILES_URL}`,
                    attribution: '© <a href="https://openstreetmap.org">OpenStreetMap</a>'
                }
            },
            layers: [
                ...mapStyle.layers,
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
            ...MAP_CONSTANTS.CONFIG
        });

        map.on('load', () => {
            setIsLoaded(true);
        });

        mapRef.current = map;

        return () => {
            map.remove();
            mapRef.current = null;
        };
    }, []);

    return { map: mapRef.current, isLoaded };
}
