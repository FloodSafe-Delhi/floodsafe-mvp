import { useEffect, useRef, useState } from 'react';
import maplibregl from 'maplibre-gl';
import { Protocol, PMTiles } from 'pmtiles';
import { MAP_CONSTANTS } from './config';
import { CITY_CONFIGS, getCurrentCity, type CityConfig } from './cityConfigs';
import mapStyle from './styles.json';

interface UseMapOptions {
    cityCode?: 'BLR' | 'DEL';
}

export function useMap(containerRef: React.RefObject<HTMLDivElement>, options?: UseMapOptions) {
    const mapRef = useRef<maplibregl.Map | null>(null);
    const [isLoaded, setIsLoaded] = useState(false);
    const [currentCity, setCurrentCity] = useState<CityConfig>(
        options?.cityCode ? CITY_CONFIGS[options.cityCode] : getCurrentCity()
    );

    useEffect(() => {
        if (!containerRef.current) return;

        // Remove existing map if city changed
        if (mapRef.current) {
            mapRef.current.remove();
            mapRef.current = null;
            setIsLoaded(false);
        }

        const protocol = new Protocol();
        maplibregl.addProtocol('pmtiles', protocol.tile);

        // Use city-specific configuration
        const cityConfig = options?.cityCode ? CITY_CONFIGS[options.cityCode] : currentCity;

        // Use the comprehensive OpenMapTiles style with flood data overlay
        const style = {
            ...mapStyle,
            sources: {
                ...mapStyle.sources,
                // Add flood visualization data with city-specific PMTiles
                'flood-tiles': {
                    type: 'vector',
                    url: `pmtiles://${cityConfig.pmtiles.flood}`,
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
            center: cityConfig.center,
            zoom: cityConfig.zoom,
            minZoom: cityConfig.minZoom,
            maxZoom: cityConfig.maxZoom,
            pitch: cityConfig.pitch,
            maxBounds: cityConfig.maxBounds,
            hash: true,
            antialias: true
        });

        map.on('load', () => {
            setIsLoaded(true);
        });

        mapRef.current = map;

        return () => {
            if (mapRef.current) {
                mapRef.current.remove();
                mapRef.current = null;
            }
        };
    }, [options?.cityCode]);

    // Method to change city programmatically
    const changeCity = (newCityCode: 'BLR' | 'DEL') => {
        setCurrentCity(CITY_CONFIGS[newCityCode]);
    };

    return {
        map: mapRef.current,
        isLoaded,
        currentCity,
        changeCity
    };
}
