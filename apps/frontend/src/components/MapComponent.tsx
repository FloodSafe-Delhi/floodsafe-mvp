import { useEffect, useRef, useState, useCallback } from 'react';
import { useMap } from '../lib/map/useMap';
import { useSensors, useReports, Sensor, Report } from '../lib/api/hooks';
import maplibregl from 'maplibre-gl';
import { Button } from './ui/button';
import { Plus, Minus, Navigation, Layers, Train, AlertCircle, MapPin } from 'lucide-react';
import MapLegend from './MapLegend';
import SearchBar from './SearchBar';
import { useCurrentCity, useCityContext } from '../contexts/CityContext';
import { isWithinCityBounds, getAvailableCities, getCityConfig, type CityKey } from '../lib/map/cityConfigs';

interface MapComponentProps {
    className?: string;
    title?: string;
    showControls?: boolean;
    showCitySelector?: boolean;
}

interface LayersVisibility {
    flood: boolean;
    sensors: boolean;
    reports: boolean;
    routes: boolean;
    metro: boolean;
}

export default function MapComponent({ className, title, showControls, showCitySelector }: MapComponentProps) {
    const mapContainer = useRef<HTMLDivElement>(null);
    const city = useCurrentCity();
    const { setCity } = useCityContext();
    const { map, isLoaded } = useMap(mapContainer, city);
    const { data: sensors } = useSensors();
    const { data: reports } = useReports();
    const [layersVisible, setLayersVisible] = useState<LayersVisibility>({
        flood: true,
        sensors: true,
        reports: true,
        routes: true,
        metro: true
    });
    const [isChangingCity, setIsChangingCity] = useState(false);
    const availableCities = showCitySelector ? getAvailableCities() : [];
    const currentCityConfig = getCityConfig(city);

    const handleCityChange = (newCity: string) => {
        setIsChangingCity(true);
        // Type-safe city change with validation
        setCity(newCity as CityKey);
        // Give the map time to reinitialize
        setTimeout(() => setIsChangingCity(false), 500);
    };

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
                    features: sensors.map((sensor: Sensor) => ({
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

        // 2. Add Community Reports Source & Layer
        if (reports && !map.getSource('reports')) {
            map.addSource('reports', {
                type: 'geojson',
                data: {
                    type: 'FeatureCollection',
                    features: reports.map((report: Report) => ({
                        type: 'Feature',
                        geometry: {
                            type: 'Point',
                            coordinates: [report.longitude, report.latitude]
                        },
                        properties: {
                            id: report.id,
                            description: report.description,
                            verified: report.verified,
                            phone_verified: report.phone_verified,
                            water_depth: report.water_depth || 'unknown',
                            vehicle_passability: report.vehicle_passability || 'unknown',
                            iot_validation_score: report.iot_validation_score,
                            timestamp: report.timestamp
                        }
                    }))
                }
            });

            // Add outer glow/halo for verified reports
            map.addLayer({
                id: 'reports-halo-layer',
                type: 'circle',
                source: 'reports',
                paint: {
                    'circle-radius': 16,
                    'circle-color': [
                        'case',
                        ['get', 'verified'], '#22c55e', // Green for verified
                        '#f59e0b' // Amber for unverified
                    ],
                    'circle-opacity': 0.2,
                    'circle-blur': 0.5
                }
            });

            // Main report markers
            map.addLayer({
                id: 'reports-layer',
                type: 'circle',
                source: 'reports',
                paint: {
                    'circle-radius': 10,
                    'circle-color': [
                        'match',
                        ['get', 'water_depth'],
                        'ankle', '#3b82f6', // Blue - low
                        'knee', '#f59e0b', // Amber - moderate
                        'waist', '#f97316', // Orange - high
                        'impassable', '#ef4444', // Red - critical
                        '#6b7280' // Gray - unknown
                    ],
                    'circle-stroke-width': 2,
                    'circle-stroke-color': [
                        'case',
                        ['get', 'verified'], '#22c55e', // Green border for verified
                        '#ffffff' // White border for unverified
                    ],
                    'circle-opacity': 0.9
                }
            });

            // Add click handler to show popup with report details
            map.on('click', 'reports-layer', (e: maplibregl.MapMouseEvent) => {
                const features = map.queryRenderedFeatures(e.point, { layers: ['reports-layer'] });
                if (!features || features.length === 0) return;

                const feature = features[0];
                // Type-safe geometry access with guard
                if (!feature.geometry || feature.geometry.type !== 'Point') return;
                const coordinates = (feature.geometry as GeoJSON.Point).coordinates.slice() as [number, number];
                const props = feature.properties;

                // Create popup HTML with safe property access
                const waterDepth = props.water_depth || 'unknown';
                const vehiclePassability = (props.vehicle_passability || 'unknown').replace('-', ' ');
                const iotScore = props.iot_validation_score ?? 0;

                const popupHTML = `
                    <div class="p-2 min-w-[200px]">
                        <div class="flex items-center gap-2 mb-2">
                            <h3 class="font-bold text-sm">Community Report</h3>
                            ${props.verified ? '<span class="text-xs bg-green-500 text-white px-2 py-0.5 rounded">✓ Verified</span>' : '<span class="text-xs bg-amber-500 text-white px-2 py-0.5 rounded">Pending</span>'}
                        </div>
                        <div class="text-xs space-y-1 text-gray-700">
                            <p><strong>Water Depth:</strong> <span class="capitalize">${waterDepth}</span></p>
                            <p><strong>Vehicle:</strong> <span class="capitalize">${vehiclePassability}</span></p>
                            <p><strong>IoT Score:</strong> ${iotScore}/100</p>
                            ${props.phone_verified ? '<p class="text-green-600">📱 Phone verified</p>' : ''}
                            <p class="text-gray-500 text-[10px] mt-2">${new Date(props.timestamp).toLocaleString()}</p>
                        </div>
                    </div>
                `;

                new maplibregl.Popup({ offset: 15 })
                    .setLngLat(coordinates)
                    .setHTML(popupHTML)
                    .addTo(map);
            });

            // Change cursor on hover
            map.on('mouseenter', 'reports-layer', () => {
                map.getCanvas().style.cursor = 'pointer';
            });

            map.on('mouseleave', 'reports-layer', () => {
                map.getCanvas().style.cursor = '';
            });
        }

        // 3. Add Safe Routes (Mock Data for Visualization)
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

        // 4. Add Pulse Effect for Critical Sensors
        // (Optional polish, can add later if needed)

    }, [map, isLoaded, sensors, reports]);

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

        // Toggle reports layers
        if (map.getLayer('reports-halo-layer')) {
            map.setLayoutProperty('reports-halo-layer', 'visibility', layersVisible.reports ? 'visible' : 'none');
        }
        if (map.getLayer('reports-layer')) {
            map.setLayoutProperty('reports-layer', 'visibility', layersVisible.reports ? 'visible' : 'none');
        }

        // Toggle routes layer
        if (map.getLayer('routes-layer')) {
            map.setLayoutProperty('routes-layer', 'visibility', layersVisible.routes ? 'visible' : 'none');
        }

        // Toggle metro layers
        if (map.getLayer('metro-lines-layer')) {
            map.setLayoutProperty('metro-lines-layer', 'visibility', layersVisible.metro ? 'visible' : 'none');
        }
        if (map.getLayer('metro-stations-layer')) {
            map.setLayoutProperty('metro-stations-layer', 'visibility', layersVisible.metro ? 'visible' : 'none');
        }
        if (map.getLayer('metro-station-names-layer')) {
            map.setLayoutProperty('metro-station-names-layer', 'visibility', layersVisible.metro ? 'visible' : 'none');
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

                    // Check if user is within current city bounds
                    const isWithinBounds = isWithinCityBounds(longitude, latitude, city);
                    const cityConfig = getCityConfig(city);

                    if (!isWithinBounds) {
                        // User is outside current city - show warning but still fly to their location
                        console.warn(`User location is outside ${cityConfig.displayName} bounds`);
                        alert(`Your location is outside the ${cityConfig.displayName} flood monitoring area. Showing your location anyway.`);
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
            ...prev,
            flood: !prev.flood
        }));
    };

    // Handle search location selection
    const handleSearchLocationSelect = useCallback((lat: number, lng: number, name: string) => {
        if (!map || !isLoaded) return;

        try {
            // Update the search-result source with the selected location
            const source = map.getSource('search-result') as maplibregl.GeoJSONSource;
            if (source) {
                source.setData({
                    type: 'FeatureCollection',
                    features: [{
                        type: 'Feature',
                        geometry: {
                            type: 'Point',
                            coordinates: [lng, lat]
                        },
                        properties: {
                            name: name
                        }
                    }]
                });
            }

            // Fly to the selected location with smooth animation
            map.flyTo({
                center: [lng, lat],
                zoom: 15,
                duration: 1500,
                essential: true
            });
        } catch (error) {
            console.log('Could not update search location:', error);
        }
    }, [map, isLoaded]);

    // Clear search marker when city changes
    useEffect(() => {
        if (!map || !isLoaded) return;

        try {
            const source = map.getSource('search-result') as maplibregl.GeoJSONSource;
            if (source) {
                source.setData({
                    type: 'FeatureCollection',
                    features: []
                });
            }
        } catch (error) {
            // Map might be in transition state during city change
            console.log('Could not clear search marker:', error);
        }
    }, [city, map, isLoaded]);

    return (
        <div className="relative w-full h-full">
            {/* Title and Search Bar - Top Left */}
            {title && (
                <div className="absolute pointer-events-auto flex flex-col gap-2" style={{ top: '16px', left: '24px', zIndex: 100 }}>
                    <div className="bg-white/90 backdrop-blur-md shadow-lg rounded-lg px-4 py-2">
                        <h1 className="text-lg font-bold text-gray-900">{title}</h1>
                        <p className="text-xs text-gray-500">Real-time flood monitoring</p>
                    </div>
                    {/* Search Bar below title */}
                    {showCitySelector && (
                        <SearchBar
                            onLocationSelect={handleSearchLocationSelect}
                            cityKey={city}
                            placeholder={`Search in ${currentCityConfig.displayName}...`}
                            className="w-72"
                        />
                    )}
                </div>
            )}

            {/* City Selector - Top Right */}
            {showCitySelector && availableCities.length > 0 && (
                <div className="absolute top-4 right-4" style={{ zIndex: 60 }}>
                    <div className="bg-white/90 backdrop-blur-sm shadow-xl rounded-lg px-3 py-2 border border-gray-200">
                        <div className="flex items-center gap-2">
                            <MapPin className="w-4 h-4 text-blue-600 flex-shrink-0" />
                            <select
                                id="city-selector"
                                name="city"
                                value={city}
                                onChange={(e) => handleCityChange(e.target.value)}
                                disabled={isChangingCity}
                                className="bg-transparent text-gray-900 font-semibold text-sm border-none focus:outline-none focus:ring-0 cursor-pointer pr-6 disabled:opacity-50"
                            >
                                {availableCities.map((cityKey) => {
                                    const config = getCityConfig(cityKey);
                                    return (
                                        <option key={cityKey} value={cityKey}>
                                            {config.displayName}
                                        </option>
                                    );
                                })}
                            </select>
                        </div>
                    </div>
                </div>
            )}
            {isChangingCity && (
                <div className="absolute inset-0 bg-white/80 backdrop-blur-sm flex items-center justify-center" style={{ zIndex: 90 }}>
                    <div className="bg-white shadow-xl rounded-lg p-6 flex flex-col items-center gap-3">
                        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
                        <p className="text-sm font-medium text-gray-700">
                            Loading {currentCityConfig.displayName} flood atlas...
                        </p>
                    </div>
                </div>
            )}
            <div ref={mapContainer} className={`${className} relative`} style={{ width: '100%', height: '100%', minHeight: '300px', zIndex: 1 }} />

            {/* Map Controls Overlay */}
            {showControls && isLoaded && (
                <>
                    {/* Zoom Controls - Bottom Right */}
                    <div className="absolute right-4 flex flex-col gap-2" style={{ bottom: '144px', zIndex: 60 }}>
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
                        <Button
                            size="icon"
                            onClick={() => setLayersVisible(prev => ({ ...prev, metro: !prev.metro }))}
                            className={`${layersVisible.metro ? '!bg-indigo-500 !hover:bg-indigo-600 !text-white' : '!bg-white !hover:bg-gray-100 !text-gray-800 border-2 border-gray-300'} shadow-xl rounded-full w-11 h-11 !opacity-100`}
                            title="Toggle metro routes"
                        >
                            <Train className="h-5 w-5" />
                        </Button>
                        <Button
                            size="icon"
                            onClick={() => setLayersVisible(prev => ({ ...prev, reports: !prev.reports }))}
                            className={`${layersVisible.reports ? '!bg-orange-500 !hover:bg-orange-600 !text-white' : '!bg-white !hover:bg-gray-100 !text-gray-800 border-2 border-gray-300'} shadow-xl rounded-full w-11 h-11 !opacity-100`}
                            title="Toggle community reports"
                        >
                            <AlertCircle className="h-5 w-5" />
                        </Button>
                    </div>

                    {/* Map Legend - Bottom Left */}
                    <div className="absolute" style={{ bottom: '144px', left: '24px', zIndex: 60 }}>
                        <MapLegend className="max-w-xs" />
                    </div>
                </>
            )}
        </div>
    );
}
