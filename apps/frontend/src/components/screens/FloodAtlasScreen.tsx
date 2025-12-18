import { useState, useEffect } from 'react';
import MapComponent from '../MapComponent';
import { Button } from '../ui/button';
import { Navigation } from 'lucide-react';
import { NavigationPanel } from '../NavigationPanel';
import { useCurrentCity } from '../../contexts/CityContext';
import { useAuth } from '../../contexts/AuthContext';
import type { RouteOption, MetroStation } from '../../types';
import { toast } from 'sonner';

interface FloodAtlasScreenProps {
    initialDestination?: [number, number] | null;
    onClearInitialDestination?: () => void;
}

export function FloodAtlasScreen({
    initialDestination,
    onClearInitialDestination
}: FloodAtlasScreenProps) {
    const city = useCurrentCity();
    const { user: _user } = useAuth();

    // Navigation state
    const [showNavigationPanel, setShowNavigationPanel] = useState(!!initialDestination);
    const [navigationRoutes, setNavigationRoutes] = useState<RouteOption[]>([]);
    const [selectedRouteId, setSelectedRouteId] = useState<string | null>(null);
    const [navigationOrigin, setNavigationOrigin] = useState<{ lat: number; lng: number } | null>(null);
    const [navigationDestination, setNavigationDestination] = useState<{ lat: number; lng: number } | null>(null);
    const [nearbyMetros, _setNearbyMetros] = useState<MetroStation[]>([]);
    const [floodZones, setFloodZones] = useState<GeoJSON.FeatureCollection | undefined>(undefined);
    const [userLocation, setUserLocation] = useState<{ lat: number; lng: number } | null>(null);

    // Geolocation - get user's current location with retry mechanism
    useEffect(() => {
        if (!('geolocation' in navigator)) {
            // Browser doesn't support geolocation - use default
            const fallbackCoords = city === 'bangalore'
                ? { lat: 12.9716, lng: 77.5946 }
                : { lat: 28.6139, lng: 77.2090 };
            setUserLocation(fallbackCoords);
            setNavigationOrigin(fallbackCoords);
            return;
        }

        const setLocationFromPosition = (position: GeolocationPosition) => {
            const loc = { lat: position.coords.latitude, lng: position.coords.longitude };
            setUserLocation(loc);
            setNavigationOrigin(loc);
        };

        const useFallback = () => {
            const fallbackCoords = city === 'bangalore'
                ? { lat: 12.9716, lng: 77.5946 }
                : { lat: 28.6139, lng: 77.2090 };
            setUserLocation(fallbackCoords);
            setNavigationOrigin(fallbackCoords);
        };

        // Try with coarse location first (faster), then retry with high accuracy if needed
        navigator.geolocation.getCurrentPosition(
            setLocationFromPosition,
            (error) => {
                if (error.code === error.TIMEOUT) {
                    // Timeout: Retry with lower accuracy and longer timeout
                    console.warn('Geolocation timeout, retrying with lower accuracy...');
                    navigator.geolocation.getCurrentPosition(
                        setLocationFromPosition,
                        (retryError) => {
                            console.warn('Geolocation retry failed:', retryError);
                            useFallback();
                        },
                        { enableHighAccuracy: false, timeout: 15000, maximumAge: 300000 }
                    );
                } else {
                    console.warn('Geolocation error:', error.message);
                    useFallback();
                }
            },
            { enableHighAccuracy: false, timeout: 15000, maximumAge: 60000 }
        );
    }, [city]);

    // Handle initial destination from HomeScreen (when user clicks "Alt Routes" on an alert)
    useEffect(() => {
        if (initialDestination) {
            setShowNavigationPanel(true);
            // initialDestination is [lng, lat] from alert.coordinates
            setNavigationDestination({ lat: initialDestination[1], lng: initialDestination[0] });
            toast.info('Opening navigation with destination from alert');
        }
    }, [initialDestination]);

    const handleRoutesCalculated = (routes: RouteOption[], zones: GeoJSON.FeatureCollection) => {
        setNavigationRoutes(routes);
        setFloodZones(zones);
        if (routes.length > 0) {
            setSelectedRouteId(routes[0].id); // Auto-select first route
        }
        // Clear the initial destination after routes are calculated
        onClearInitialDestination?.();
    };

    const handleRouteSelected = (route: RouteOption) => {
        setSelectedRouteId(route.id);
    };

    const handleMetroSelected = (station: MetroStation) => {
        // When user selects a metro station, set it as destination
        setNavigationDestination({ lat: station.lat, lng: station.lng });
        toast.success(`Selected ${station.name} as destination`);
    };

    return (
        <div className="fixed inset-0 top-14 md:top-0 bottom-0 bg-transparent">
            <MapComponent
                className="w-full h-full"
                title="Flood Atlas"
                showControls={true}
                showCitySelector={true}
                navigationRoutes={navigationRoutes}
                selectedRouteId={selectedRouteId ?? undefined}
                navigationOrigin={navigationOrigin ?? undefined}
                navigationDestination={navigationDestination ?? undefined}
                nearbyMetros={nearbyMetros}
                floodZones={floodZones}
                onMetroClick={handleMetroSelected}
            />

            {/* Floating Route Button - Only show when panel is closed */}
            {!showNavigationPanel && (
                <div
                    className="fixed right-4 pointer-events-auto"
                    style={{ bottom: '80px', zIndex: 9999 }}
                >
                    <Button
                        onClick={() => setShowNavigationPanel(true)}
                        className="shadow-xl bg-blue-600 hover:bg-blue-700 text-white"
                        size="lg"
                    >
                        <Navigation className="mr-2 h-5 w-5" />
                        Plan Safe Route
                    </Button>
                </div>
            )}

            {/* Navigation Panel */}
            <NavigationPanel
                isOpen={showNavigationPanel}
                onClose={() => setShowNavigationPanel(false)}
                userLocation={userLocation}
                city={city}
                onRoutesCalculated={handleRoutesCalculated}
                onRouteSelected={handleRouteSelected}
                onMetroSelected={handleMetroSelected}
                onOriginChange={setNavigationOrigin}
                onDestinationChange={setNavigationDestination}
                initialDestination={navigationDestination}
            />
        </div>
    );
}
