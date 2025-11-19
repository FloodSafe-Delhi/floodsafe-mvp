import { useEffect, useRef, useState } from 'react';
import maplibregl from 'maplibre-gl';
import { Protocol, PMTiles } from 'pmtiles';
import { MAP_CONSTANTS } from './config';

export function useMap(containerRef: React.RefObject<HTMLDivElement>) {
    const mapRef = useRef<maplibregl.Map | null>(null);
    const [isLoaded, setIsLoaded] = useState(false);

    useEffect(() => {
        if (!containerRef.current || mapRef.current) return;

        const protocol = new Protocol();
        maplibregl.addProtocol('pmtiles', protocol.tile);

        const map = new maplibregl.Map({
            container: containerRef.current,
            style: {
                version: 8,
                sources: {
                    'pmtiles-source': {
                        type: 'vector',
                        url: `pmtiles://${MAP_CONSTANTS.PMTILES_URL}`,
                        attribution: '© <a href="https://openstreetmap.org">OpenStreetMap</a>'
                    },
                    'basemap-source': {
                        type: 'vector',
                        url: `pmtiles://${MAP_CONSTANTS.BASEMAP_URL}`,
                    }
                },
                layers: [
                    // Simplified style for demonstration; real style should be imported from styles.json
                    {
                        id: 'background',
                        type: 'background',
                        paint: { 'background-color': '#f0f0f0' }
                    },
                    {
                        id: 'pmtiles-layer',
                        type: 'fill',
                        source: 'pmtiles-source',
                        'source-layer': 'stream_influence_water_difference',
                        paint: {
                            'fill-color': '#519ea2',
                            'fill-opacity': 0.8
                        }
                    }
                ]
            },
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
