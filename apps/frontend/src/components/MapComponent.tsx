import React, { useEffect, useRef, useState } from 'react';
import { useMap } from '../lib/map/useMap';
import { useSensors } from '../lib/api/hooks';
import maplibregl from 'maplibre-gl';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Plus, Minus, Navigation, Layers } from 'lucide-react';
import MapLegend from './MapLegend';

interface MapComponentProps {
    className?: string;
    title?: string;
    showControls?: boolean;
}

export default function MapComponent({ className, title, showControls }: MapComponentProps) {
    const mapContainer = useRef<HTMLDivElement>(null);
    const { map, isLoaded } = useMap(mapContainer);
    const { data: sensors } = useSensors();
    const [layersVisible, setLayersVisible] = useState({
        flood: true,
        sensors: true,
        routes: true
    });

    // Force resize when the component mounts or className changes
    useEffect(() => {
        if (map) {
            map.resize();
        }
    }, [map, className]);

    useEffect(() => {
        if (!map || !isLoaded) return;

        // 1. Add Sensors Source & Layer (Existing)
        if (sensors && !map.getSource('sensors')) {
            map.addSource('sensors', {
                type: 'geojson',
                data: {
                    type: 'FeatureCollection',
                    features: sensors.map(sensor => ({
                        type: 'Feature',
                        geometry: {
                            type: 'Point',
                            coordinates: [sensor.longitude, sensor.latitude]
                        },
                        properties: {
                            id: sensor.id,
                            status: sensor.status,
                            last_ping: sensor.last_ping
                        }
                    }))
                }
            });

            map.addLayer({
                id: 'sensors-layer',
                type: 'circle',
                source: 'sensors',
                paint: {
                    'circle-radius': 8,
                    'circle-color': [
                        'match',
                        ['get', 'status'],
                        'active', '#22c55e', // Green
                        'warning', '#f97316', // Orange
                        'critical', '#ef4444', // Red
                        '#9ca3af' // Gray default
                    ],
                    'circle-stroke-width': 2,
                    'circle-stroke-color': '#ffffff'
                }
            });
        }

        // 2. Add Safe Routes (Mock Data for Visualization)
        if (!map.getSource('safe-routes')) {
            map.addSource('safe-routes', {
                type: 'geojson',
                data: {
                    type: 'FeatureCollection',
                    features: [
                        {
                            type: 'Feature',
                            properties: { type: 'safe' },
                            geometry: {
                                type: 'LineString',
                                coordinates: [
                                    [77.5777, 12.9776],
                                    [77.5800, 12.9800],
                                    [77.5850, 12.9820]
                                ]
                            }
                        },
                        {
                            type: 'Feature',
                            properties: { type: 'flooded' },
                            geometry: {
                                type: 'LineString',
                                coordinates: [
                                    [77.5700, 12.9700],
                                    [77.5720, 12.9720],
                                    [77.5750, 12.9750]
                                ]
                            }
                        }
                    ]
                }
            });

            map.addLayer({
                id: 'routes-layer',
                type: 'line',
                source: 'safe-routes',
                layout: {
                    'line-join': 'round',
                    'line-cap': 'round'
                },
                paint: {
                    'line-color': [
                        'match',
                        ['get', 'type'],
                        'safe', '#22c55e', // Green
                        'flooded', '#ef4444', // Red
                        '#888888'
                    ],
                    'line-width': 4,
                    'line-opacity': 0.8
                }
            });
        }

        // 3. Add Pulse Effect for Critical Sensors
        // (Optional polish, can add later if needed)

    }, [map, isLoaded, sensors]);

    // Toggle layer visibility
    useEffect(() => {
        if (!map || !isLoaded) return;

        // Toggle flood layer
        if (map.getLayer('flood-layer')) {
            map.setLayoutProperty('flood-layer', 'visibility', layersVisible.flood ? 'visible' : 'none');
        }

        // Toggle sensors layer
        if (map.getLayer('sensors-layer')) {
            map.setLayoutProperty('sensors-layer', 'visibility', layersVisible.sensors ? 'visible' : 'none');
        }

        // Toggle routes layer
        if (map.getLayer('routes-layer')) {
            map.setLayoutProperty('routes-layer', 'visibility', layersVisible.routes ? 'visible' : 'none');
        }
    }, [map, isLoaded, layersVisible]);

    const handleZoomIn = () => {
        if (map) map.zoomIn();
    };

    const handleZoomOut = () => {
        if (map) map.zoomOut();
    };

    const handleMyLocation = () => {
        if (!map) return;

        if ('geolocation' in navigator) {
            navigator.geolocation.getCurrentPosition(
                (position) => {
                    const { longitude, latitude } = position.coords;

                    // Bangalore bounds (approximate)
                    const bangaloreBounds = {
                        minLng: 77.199861111,
                        maxLng: 77.899861111,
                        minLat: 12.600138889,
                        maxLat: 13.400138889
                    };

                    const isInBangalore =
                        longitude >= bangaloreBounds.minLng &&
                        longitude <= bangaloreBounds.maxLng &&
                        latitude >= bangaloreBounds.minLat &&
                        latitude <= bangaloreBounds.maxLat;

                    if (!isInBangalore) {
                        // User is outside Bangalore - show warning but still fly to their location
                        console.warn('User location is outside Bangalore bounds');
                        alert('Your location is outside the Bangalore flood monitoring area. Showing your location anyway.');
                    }

                    map.flyTo({
                        center: [longitude, latitude],
                        zoom: 14,
                        duration: 2000
                    });

                    // Add a marker at user's location
                    new maplibregl.Marker({ color: '#3b82f6' })
                        .setLngLat([longitude, latitude])
                        .addTo(map);
                },
                (error) => {
                    console.error('Error getting location:', error);
                    alert('Unable to get your location. Please enable location permissions.');
                }
            );
        } else {
            alert('Geolocation is not supported by your browser.');
        }
    };

    const toggleLayers = () => {
        setLayersVisible(prev => ({
            flood: !prev.flood,
            sensors: prev.sensors,
            routes: prev.routes
        }));
    };

    return (
        <div className="relative w-full h-full">
            {title && (
                <div className="absolute top-4 left-4 right-4 z-10 flex justify-between items-start pointer-events-none">
                    <div className="bg-white/90 backdrop-blur-md shadow-lg rounded-lg px-4 py-2 pointer-events-auto">
                        <h1 className="text-lg font-bold text-gray-900">{title}</h1>
                        <p className="text-xs text-gray-500">Real-time flood monitoring</p>
                    </div>

                    <div className="pointer-events-auto">
                        <Badge variant="secondary" className="bg-white shadow">
                            Online
                        </Badge>
                    </div>
                </div>
            )}
            <div ref={mapContainer} className={className} style={{ width: '100%', height: '100%', minHeight: '300px' }} />

            {/* Map Controls Overlay */}
            {showControls && isLoaded && (
                <>
                    {/* Zoom Controls - Bottom Right */}
                    <div className="absolute right-4 flex flex-col gap-2 z-[60]" style={{ bottom: '144px' }}>
                        <Button
                            size="icon"
                            onClick={handleZoomIn}
                            className="!bg-white !hover:bg-gray-100 !text-gray-800 shadow-xl rounded-full w-11 h-11 border-2 border-gray-300 !opacity-100"
                            title="Zoom in"
                        >
                            <Plus className="h-5 w-5" />
                        </Button>
                        <Button
                            size="icon"
                            onClick={handleZoomOut}
                            className="!bg-white !hover:bg-gray-100 !text-gray-800 shadow-xl rounded-full w-11 h-11 border-2 border-gray-300 !opacity-100"
                            title="Zoom out"
                        >
                            <Minus className="h-5 w-5" />
                        </Button>
                        <Button
                            size="icon"
                            onClick={handleMyLocation}
                            className="!bg-blue-500 !hover:bg-blue-600 !text-white shadow-xl rounded-full w-11 h-11 !opacity-100"
                            title="My location"
                        >
                            <Navigation className="h-5 w-5" />
                        </Button>
                        <Button
                            size="icon"
                            onClick={toggleLayers}
                            className={`${layersVisible.flood ? '!bg-green-500 !hover:bg-green-600 !text-white' : '!bg-white !hover:bg-gray-100 !text-gray-800 border-2 border-gray-300'} shadow-xl rounded-full w-11 h-11 !opacity-100`}
                            title="Toggle flood layer"
                        >
                            <Layers className="h-5 w-5" />
                        </Button>
                    </div>

                    {/* Map Legend - Bottom Left */}
                    <div className="absolute z-[60]" style={{ bottom: '144px', left: '24px' }}>
                        <MapLegend className="max-w-xs" />
                    </div>
                </>
            )}
        </div>
    );
}
