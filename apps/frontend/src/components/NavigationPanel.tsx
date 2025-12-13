import React, { useState, useEffect } from 'react';
import { Navigation, MapPin, Clock, Shield, Bike, Car, Footprints, Train, Bookmark, Star, Trash2, LocateFixed, GitCompare, X, Loader2 } from 'lucide-react';
import { Sheet, SheetContent, SheetHeader, SheetTitle } from './ui/sheet';
import { Button } from './ui/button';
import SmartSearchBar from './SmartSearchBar';
import { useCompareRoutes, useNearbyMetros, useSavedRoutes, useCreateSavedRoute, useDeleteSavedRoute, useIncrementRouteUsage } from '../lib/api/hooks';
import { RouteOption, MetroStation, RouteComparisonResponse } from '../types';
import { toast } from 'sonner';
import { useAuth } from '../contexts/AuthContext';
import { RouteComparisonCard } from './RouteComparisonCard';

interface NavigationPanelProps {
    isOpen: boolean;
    onClose: () => void;
    userLocation: { lat: number; lng: number } | null;
    city: 'bangalore' | 'delhi';
    onRoutesCalculated: (routes: RouteOption[], floodZones: GeoJSON.FeatureCollection) => void;
    onRouteSelected: (route: RouteOption) => void;
    onMetroSelected: (station: MetroStation) => void;
    onOriginChange?: (origin: { lat: number; lng: number } | null) => void;
    onDestinationChange?: (destination: { lat: number; lng: number } | null) => void;
    initialDestination?: { lat: number; lng: number; name?: string } | null;
}

export function NavigationPanel({
    isOpen,
    onClose,
    userLocation,
    city,
    onRoutesCalculated,
    onRouteSelected,
    onMetroSelected,
    onOriginChange,
    onDestinationChange,
    initialDestination,
}: NavigationPanelProps) {
    const { user } = useAuth();
    const [origin, setOrigin] = useState<{ lat: number; lng: number; name: string } | null>(null);
    const [destination, setDestination] = useState<{ lat: number; lng: number; name: string } | null>(null);
    const [useCurrentLocation, setUseCurrentLocation] = useState(true);
    const [mode, setMode] = useState<'driving' | 'walking' | 'cycling'>('driving');
    const [_routes, setRoutes] = useState<RouteOption[]>([]);
    const [_selectedRouteId, setSelectedRouteId] = useState<string | null>(null);
    const [avoidMLRisk, setAvoidMLRisk] = useState(false);
    const [comparison, setComparison] = useState<RouteComparisonResponse | null>(null);
    const [selectedRouteType, setSelectedRouteType] = useState<'normal' | 'floodsafe' | null>(null);

    // Set origin from userLocation when using current location
    useEffect(() => {
        if (useCurrentLocation && userLocation) {
            setOrigin({
                lat: userLocation.lat,
                lng: userLocation.lng,
                name: 'Current Location'
            });
        }
    }, [userLocation, useCurrentLocation]);

    // Set destination from initialDestination prop (when coming from "Alt Routes" button)
    useEffect(() => {
        if (initialDestination && isOpen) {
            setDestination({
                lat: initialDestination.lat,
                lng: initialDestination.lng,
                name: initialDestination.name || `Location (${initialDestination.lat.toFixed(4)}, ${initialDestination.lng.toFixed(4)})`
            });
        }
    }, [initialDestination, isOpen]);

    // Notify parent of origin changes for map visualization
    useEffect(() => {
        onOriginChange?.(origin ? { lat: origin.lat, lng: origin.lng } : null);
    }, [origin, onOriginChange]);

    // Notify parent of destination changes for map visualization
    useEffect(() => {
        onDestinationChange?.(destination ? { lat: destination.lat, lng: destination.lng } : null);
    }, [destination, onDestinationChange]);

    const { mutate: compareRoutes, isPending: isCalculating } = useCompareRoutes();
    const { data: metrosData } = useNearbyMetros(
        origin?.lat ?? null,
        origin?.lng ?? null,
        city === 'bangalore' ? 'BLR' : 'DEL'
    );

    const metros = metrosData?.metros ?? [];

    // Saved routes
    const { data: savedRoutes = [] } = useSavedRoutes(user?.id);
    const { mutate: createSavedRoute, isPending: isSaving } = useCreateSavedRoute();
    const { mutate: deleteSavedRoute } = useDeleteSavedRoute();
    const { mutate: incrementUsage } = useIncrementRouteUsage();

    const handleOriginSelect = (lat: number, lng: number, name: string) => {
        setOrigin({ lat, lng, name });
        setUseCurrentLocation(false);
    };

    const handleUseCurrentLocation = () => {
        if (userLocation) {
            setOrigin({
                lat: userLocation.lat,
                lng: userLocation.lng,
                name: 'Current Location'
            });
            setUseCurrentLocation(true);
            toast.success('Using current location as starting point');
        } else {
            toast.error('Unable to detect your location');
        }
    };

    const handleDestinationSelect = (lat: number, lng: number, name: string) => {
        setDestination({ lat, lng, name });
    };

    const handleFindRoutes = () => {
        if (!origin) {
            toast.error('Please select a starting location');
            return;
        }

        if (!destination) {
            toast.error('Please select a destination');
            return;
        }

        compareRoutes(
            {
                origin: { lat: origin.lat, lng: origin.lng },
                destination: { lat: destination.lat, lng: destination.lng },
                mode,
                city: city === 'bangalore' ? 'BLR' : 'DEL',
            },
            {
                onSuccess: (data) => {
                    setComparison(data);
                    setSelectedRouteType(null);

                    // Build routes array for map display from comparison
                    const routesForMap: RouteOption[] = [];
                    if (data.floodsafe_route) {
                        routesForMap.push(data.floodsafe_route);
                    }
                    setRoutes(routesForMap);

                    // Pass routes and flood zones to parent
                    if (routesForMap.length > 0) {
                        onRoutesCalculated(routesForMap, data.flood_zones);
                        toast.success('Route comparison ready');
                    } else {
                        toast.error('No routes found');
                    }
                },
                onError: (error) => {
                    toast.error('Failed to calculate routes');
                    console.error('Route calculation error:', error);
                },
            }
        );
    };

    const _handleRouteSelect = (route: RouteOption) => {
        setSelectedRouteId(route.id);
        onRouteSelected(route);
    };

    // Handle normal route selection from comparison card
    const handleSelectNormalRoute = () => {
        if (comparison?.normal_route) {
            setSelectedRouteType('normal');
            // Convert NormalRouteOption to RouteOption format for map display
            const normalAsRouteOption: RouteOption = {
                id: comparison.normal_route.id,
                type: 'fast',
                city_code: city === 'bangalore' ? 'BLR' : 'DEL',
                geometry: comparison.normal_route.geometry,
                distance_meters: comparison.normal_route.distance_meters,
                duration_seconds: comparison.normal_route.duration_seconds,
                safety_score: comparison.normal_route.safety_score,
                risk_level: comparison.normal_route.safety_score >= 70 ? 'low' : comparison.normal_route.safety_score >= 40 ? 'medium' : 'high',
                flood_intersections: comparison.normal_route.flood_intersections,
                instructions: comparison.normal_route.instructions,
            };
            onRouteSelected(normalAsRouteOption);
            toast.info('Showing normal (fastest) route');
        }
    };

    // Handle FloodSafe route selection from comparison card
    const handleSelectFloodSafeRoute = () => {
        if (comparison?.floodsafe_route) {
            setSelectedRouteType('floodsafe');
            onRouteSelected(comparison.floodsafe_route);
            toast.success('Showing FloodSafe route');
        }
    };

    const handleMetroSelect = (station: MetroStation) => {
        onMetroSelected(station);
        toast.success(`Showing route to ${station.name}`);
    };

    const handleSaveRoute = () => {
        if (!user || !origin || !destination) {
            toast.error('Please select both origin and destination first');
            return;
        }

        createSavedRoute({
            user_id: user.id,
            name: `${origin.name} → ${destination.name}`,
            origin_latitude: origin.lat,
            origin_longitude: origin.lng,
            origin_name: origin.name,
            destination_latitude: destination.lat,
            destination_longitude: destination.lng,
            destination_name: destination.name,
            transport_mode: mode,
        }, {
            onSuccess: () => {
                toast.success('Route saved to favorites!');
            },
            onError: () => {
                toast.error('Failed to save route');
            }
        });
    };

    const handleLoadSavedRoute = (saved: any) => {
        // Set origin from saved route
        if (saved.origin_latitude && saved.origin_longitude) {
            setOrigin({
                lat: saved.origin_latitude,
                lng: saved.origin_longitude,
                name: saved.origin_name || 'Saved Origin'
            });
            setUseCurrentLocation(false);
        }

        // Set destination from saved route
        setDestination({
            lat: saved.destination_latitude,
            lng: saved.destination_longitude,
            name: saved.destination_name || saved.name
        });
        setMode(saved.transport_mode as 'driving' | 'walking' | 'cycling');

        // Increment usage count
        incrementUsage(saved.id);

        toast.info(`Loaded: ${saved.name}`);
    };

    const handleDeleteSavedRoute = (routeId: string, routeName: string) => {
        deleteSavedRoute(routeId, {
            onSuccess: () => {
                toast.success(`Deleted "${routeName}"`);
            },
            onError: () => {
                toast.error('Failed to delete route');
            }
        });
    };

    const formatDistance = (meters: number) => {
        if (meters < 1000) return `${Math.round(meters)}m`;
        return `${(meters / 1000).toFixed(1)}km`;
    };

    const _formatDuration = (seconds?: number) => {
        if (!seconds) return 'N/A';
        const minutes = Math.round(seconds / 60);
        if (minutes < 60) return `${minutes}min`;
        const hours = Math.floor(minutes / 60);
        const mins = minutes % 60;
        return `${hours}h ${mins}min`;
    };

    const _getSafetyColor = (score: number) => {
        if (score >= 75) return 'text-green-600';
        if (score >= 50) return 'text-yellow-600';
        if (score >= 25) return 'text-orange-600';
        return 'text-red-600';
    };

    const _getSafetyLabel = (score: number) => {
        if (score >= 75) return 'Safe';
        if (score >= 50) return 'Moderate';
        if (score >= 25) return 'Caution';
        return 'Unsafe';
    };

    const _getRouteTypeIcon = (type: string) => {
        switch (type) {
            case 'safe': return <Shield className="h-4 w-4 text-green-600" />;
            case 'balanced': return <Navigation className="h-4 w-4 text-blue-600" />;
            case 'fast': return <Clock className="h-4 w-4 text-orange-600" />;
            default: return <Navigation className="h-4 w-4" />;
        }
    };

    return (
        <Sheet open={isOpen} onOpenChange={(open) => { if (!open) onClose(); }}>
            <SheetContent side="bottom" className="h-[80vh] overflow-y-auto">
                <SheetHeader className="relative pb-2 border-b">
                    <SheetTitle className="flex items-center gap-2 pr-8">
                        <Navigation className="h-5 w-5 text-blue-600" />
                        Plan Safe Route
                    </SheetTitle>
                    {/* Explicit close button for better visibility on bottom sheet */}
                    <button
                        onClick={onClose}
                        className="absolute top-0 right-0 p-2 rounded-full hover:bg-gray-100 transition-colors"
                        aria-label="Close navigation panel"
                    >
                        <X className="h-5 w-5 text-gray-500" />
                    </button>
                </SheetHeader>

                <div className="mt-6 space-y-6">
                    {/* Starting Location */}
                    <div className="space-y-2">
                        <div className="flex items-center justify-between">
                            <label className="text-sm font-medium">Starting Location</label>
                            <Button
                                variant="ghost"
                                size="sm"
                                onClick={handleUseCurrentLocation}
                                className={`text-xs h-7 ${useCurrentLocation ? 'text-green-600' : 'text-gray-500'}`}
                            >
                                <LocateFixed className="h-3 w-3 mr-1" />
                                Use GPS
                            </Button>
                        </div>
                        <SmartSearchBar
                            placeholder="Search for starting point..."
                            onLocationSelect={handleOriginSelect}
                        />
                        {origin && (
                            <div className="flex items-center gap-2 p-2 bg-green-50 rounded-lg border border-green-200">
                                <MapPin className="h-4 w-4 text-green-600 flex-shrink-0" />
                                <span className="text-sm text-green-800 truncate">
                                    {origin.name}
                                </span>
                            </div>
                        )}
                    </div>

                    {/* Destination Search */}
                    <div className="space-y-2">
                        <label className="text-sm font-medium">Destination</label>
                        <SmartSearchBar
                            placeholder="Search for destination..."
                            onLocationSelect={handleDestinationSelect}
                        />
                        {destination && (
                            <div className="flex items-center gap-2 p-2 bg-red-50 rounded-lg border border-red-200">
                                <MapPin className="h-4 w-4 text-red-600 flex-shrink-0" />
                                <span className="text-sm text-red-800 truncate">
                                    {destination.name}
                                </span>
                            </div>
                        )}
                    </div>

                    {/* Transport Mode Selector */}
                    <div className="space-y-2">
                        <label className="text-sm font-medium">Transport Mode</label>
                        <div className="flex gap-2">
                            <Button
                                variant={mode === 'driving' ? 'default' : 'outline'}
                                size="sm"
                                onClick={() => setMode('driving')}
                                className="flex-1"
                            >
                                <Car className="h-4 w-4 mr-2" />
                                Drive
                            </Button>
                            <Button
                                variant={mode === 'walking' ? 'default' : 'outline'}
                                size="sm"
                                onClick={() => setMode('walking')}
                                className="flex-1"
                            >
                                <Footprints className="h-4 w-4 mr-2" />
                                Walk
                            </Button>
                            <Button
                                variant={mode === 'cycling' ? 'default' : 'outline'}
                                size="sm"
                                onClick={() => setMode('cycling')}
                                className="flex-1"
                            >
                                <Bike className="h-4 w-4 mr-2" />
                                Bike
                            </Button>
                        </div>
                    </div>

                    {/* ML Risk Toggle (Placeholder) */}
                    <div className="flex items-center gap-2 p-3 bg-gray-50 rounded-lg opacity-50">
                        <input
                            type="checkbox"
                            disabled
                            checked={avoidMLRisk}
                            onChange={(e) => setAvoidMLRisk(e.target.checked)}
                            className="rounded"
                        />
                        <span className="text-sm text-gray-600">
                            Avoid AI-predicted flood zones (coming soon)
                        </span>
                    </div>

                    {/* Find Routes Button */}
                    <Button
                        onClick={handleFindRoutes}
                        disabled={!origin || !destination || isCalculating}
                        className="w-full"
                        size="lg"
                    >
                        {isCalculating ? (
                            <>
                                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                                Calculating Routes...
                            </>
                        ) : (
                            <>
                                <Navigation className="h-4 w-4 mr-2" />
                                Find Safe Routes
                            </>
                        )}
                    </Button>

                    {/* Route Comparison Results */}
                    {comparison && (
                        <div className="space-y-3">
                            <h3 className="font-medium flex items-center gap-2">
                                <GitCompare className="h-4 w-4" />
                                Route Comparison
                            </h3>
                            <RouteComparisonCard
                                comparison={comparison}
                                onSelectNormal={handleSelectNormalRoute}
                                onSelectFloodSafe={handleSelectFloodSafeRoute}
                                selectedRoute={selectedRouteType}
                            />
                        </div>
                    )}

                    {/* Nearby Metro Stations */}
                    {metros.length > 0 && (
                        <div className="space-y-3">
                            <h3 className="font-medium">Nearby Metro Stations</h3>
                            <div className="space-y-2">
                                {metros.slice(0, 5).map((station) => (
                                    <button
                                        key={station.id}
                                        onClick={() => handleMetroSelect(station)}
                                        className="w-full p-3 rounded-lg border border-gray-200 hover:border-gray-300 text-left transition-all"
                                    >
                                        <div className="flex items-start justify-between">
                                            <div className="flex items-start gap-2">
                                                <Train
                                                    className="h-4 w-4 mt-0.5"
                                                    style={{ color: station.color }}
                                                />
                                                <div>
                                                    <div className="font-medium">{station.name}</div>
                                                    <div className="text-xs text-gray-600">{station.line}</div>
                                                </div>
                                            </div>
                                            <div className="text-right text-sm">
                                                <div className="font-medium">{formatDistance(station.distance_meters)}</div>
                                                <div className="text-xs text-gray-600">
                                                    {station.walking_minutes} min walk
                                                </div>
                                            </div>
                                        </div>
                                    </button>
                                ))}
                            </div>
                        </div>
                    )}

                    {/* Saved Routes */}
                    {savedRoutes.length > 0 && (
                        <div className="space-y-3">
                            <h3 className="font-medium flex items-center gap-2">
                                <Star className="h-4 w-4 text-yellow-500" />
                                Saved Routes
                            </h3>
                            <div className="space-y-2">
                                {savedRoutes.slice(0, 5).map((saved) => (
                                    <div
                                        key={saved.id}
                                        className="w-full p-3 rounded-lg border border-gray-200 hover:border-blue-300 transition-all"
                                    >
                                        <div className="flex items-center justify-between">
                                            <button
                                                onClick={() => handleLoadSavedRoute(saved)}
                                                className="flex items-center gap-2 flex-1 text-left"
                                            >
                                                <Bookmark className="h-4 w-4 text-blue-500 flex-shrink-0" />
                                                <div className="flex-1 min-w-0">
                                                    <div className="font-medium text-sm truncate">{saved.name}</div>
                                                    <div className="text-xs text-gray-500">
                                                        Used {saved.use_count}x · {saved.transport_mode}
                                                    </div>
                                                </div>
                                            </button>
                                            <button
                                                onClick={(e) => {
                                                    e.stopPropagation();
                                                    handleDeleteSavedRoute(saved.id, saved.name);
                                                }}
                                                className="p-2 text-gray-400 hover:text-red-500 transition-colors"
                                                title="Delete saved route"
                                            >
                                                <Trash2 className="h-4 w-4" />
                                            </button>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}

                    {/* Save Current Route Button */}
                    {origin && destination && (
                        <Button
                            variant="outline"
                            onClick={handleSaveRoute}
                            disabled={isSaving}
                            className="w-full"
                        >
                            <Bookmark className="h-4 w-4 mr-2" />
                            {isSaving ? 'Saving...' : 'Save This Route'}
                        </Button>
                    )}
                </div>
            </SheetContent>
        </Sheet>
    );
}
