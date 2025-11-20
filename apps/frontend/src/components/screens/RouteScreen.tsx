import React, { useState, useEffect } from 'react';
import { ArrowLeft, Navigation, MapPin, AlertCircle, Zap, Shield, TrendingUp } from 'lucide-react';
import { Button } from '../ui/button';
import { Input } from '../ui/input';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../ui/card';
import { Badge } from '../ui/badge';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../ui/select';
import { useRouteCalculation, useGeocode } from '../../lib/api/hooks';
import { useCurrentCity } from '../../contexts/CityContext';
import type { LocationPoint, RouteOption, TransportMode } from '../../types';
import { toast } from 'sonner';

interface RouteScreenProps {
    onBack: () => void;
    onRouteSelected?: (route: RouteOption) => void;
}

export function RouteScreen({ onBack, onRouteSelected }: RouteScreenProps) {
    const city = useCurrentCity();
    const [originQuery, setOriginQuery] = useState('');
    const [destinationQuery, setDestinationQuery] = useState('');
    const [origin, setOrigin] = useState<LocationPoint | null>(null);
    const [destination, setDestination] = useState<LocationPoint | null>(null);
    const [mode, setMode] = useState<TransportMode>('driving');
    const [selectedRoute, setSelectedRoute] = useState<RouteOption | null>(null);
    const [showOriginResults, setShowOriginResults] = useState(false);
    const [showDestinationResults, setShowDestinationResults] = useState(false);

    const { mutate: calculateRoute, data: routeData, isPending, isError } = useRouteCalculation();
    const { data: originResults = [] } = useGeocode(originQuery, showOriginResults && originQuery.length >= 3);
    const { data: destinationResults = [] } = useGeocode(destinationQuery, showDestinationResults && destinationQuery.length >= 3);

    const handleOriginSelect = (lat: string, lon: string, displayName: string) => {
        setOrigin({ lat: parseFloat(lat), lng: parseFloat(lon) });
        setOriginQuery(displayName);
        setShowOriginResults(false);
    };

    const handleDestinationSelect = (lat: string, lon: string, displayName: string) => {
        setDestination({ lat: parseFloat(lat), lng: parseFloat(lon) });
        setDestinationQuery(displayName);
        setShowDestinationResults(false);
    };

    const handleCalculateRoute = () => {
        if (!origin || !destination) {
            toast.error('Please select both origin and destination');
            return;
        }

        calculateRoute({
            origin: { lng: origin.lng, lat: origin.lat },
            destination: { lng: destination.lng, lat: destination.lat },
            city: city === 'bangalore' ? 'BLR' : 'DEL',
            mode,
            avoid_risk_levels: ['critical', 'warning']
        }, {
            onSuccess: (data) => {
                if (data.routes && data.routes.length > 0) {
                    toast.success(`Found ${data.routes.length} route${data.routes.length > 1 ? 's' : ''}`);
                    setSelectedRoute(data.routes[0]);
                } else {
                    toast.error('No routes found');
                }
            },
            onError: (error) => {
                toast.error(`Route calculation failed: ${error.message}`);
            }
        });
    };

    const handleRouteSelection = (route: RouteOption) => {
        setSelectedRoute(route);
        if (onRouteSelected) {
            onRouteSelected(route);
        }
    };

    const formatDistance = (meters: number): string => {
        if (meters < 1000) {
            return `${Math.round(meters)}m`;
        }
        return `${(meters / 1000).toFixed(1)}km`;
    };

    const getRiskBadgeColor = (riskLevel: string) => {
        switch (riskLevel) {
            case 'low': return 'bg-green-100 text-green-800 border-green-200';
            case 'medium': return 'bg-yellow-100 text-yellow-800 border-yellow-200';
            case 'high': return 'bg-red-100 text-red-800 border-red-200';
            default: return 'bg-gray-100 text-gray-800 border-gray-200';
        }
    };

    const getRouteIcon = (type: string) => {
        switch (type) {
            case 'safe': return <Shield className="w-4 h-4" />;
            case 'fast': return <Zap className="w-4 h-4" />;
            case 'balanced': return <TrendingUp className="w-4 h-4" />;
            default: return <Navigation className="w-4 h-4" />;
        }
    };

    const getRouteLabel = (type: string) => {
        switch (type) {
            case 'safe': return 'Safest Route';
            case 'fast': return 'Fastest Route';
            case 'balanced': return 'Balanced Route';
            default: return 'Route';
        }
    };

    return (
        <div className="min-h-screen bg-gray-50 pb-20">
            {/* Header */}
            <div className="sticky top-0 z-10 bg-white border-b">
                <div className="flex items-center p-4">
                    <Button
                        variant="ghost"
                        size="icon"
                        onClick={onBack}
                        className="mr-2"
                    >
                        <ArrowLeft className="h-5 w-5" />
                    </Button>
                    <div className="flex-1">
                        <h1 className="text-xl font-bold">Safe Route Navigation</h1>
                        <p className="text-sm text-gray-600">Avoid flood-affected areas</p>
                    </div>
                    <Badge variant="outline" className="ml-2">
                        {city === 'bangalore' ? 'Bangalore' : 'Delhi'}
                    </Badge>
                </div>
            </div>

            <div className="p-4 space-y-4">
                {/* Route Input Card */}
                <Card>
                    <CardHeader>
                        <CardTitle className="text-lg">Where do you want to go?</CardTitle>
                        <CardDescription>Enter origin and destination addresses</CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                        {/* Origin Input */}
                        <div className="relative">
                            <div className="flex items-center space-x-2">
                                <MapPin className="w-5 h-5 text-blue-600" />
                                <Input
                                    placeholder="Enter starting location..."
                                    value={originQuery}
                                    onChange={(e) => {
                                        setOriginQuery(e.target.value);
                                        setShowOriginResults(true);
                                    }}
                                    onFocus={() => setShowOriginResults(true)}
                                    className="flex-1"
                                />
                            </div>
                            {showOriginResults && originResults.length > 0 && (
                                <div className="absolute z-20 w-full mt-1 bg-white border rounded-lg shadow-lg max-h-60 overflow-y-auto">
                                    {originResults.map((result, idx) => (
                                        <button
                                            key={idx}
                                            className="w-full text-left p-3 hover:bg-gray-50 border-b last:border-b-0 transition-colors"
                                            onClick={() => handleOriginSelect(result.lat, result.lon, result.display_name)}
                                        >
                                            <div className="font-medium text-sm">{result.display_name.split(',')[0]}</div>
                                            <div className="text-xs text-gray-500">{result.display_name}</div>
                                        </button>
                                    ))}
                                </div>
                            )}
                        </div>

                        {/* Destination Input */}
                        <div className="relative">
                            <div className="flex items-center space-x-2">
                                <MapPin className="w-5 h-5 text-red-600" />
                                <Input
                                    placeholder="Enter destination..."
                                    value={destinationQuery}
                                    onChange={(e) => {
                                        setDestinationQuery(e.target.value);
                                        setShowDestinationResults(true);
                                    }}
                                    onFocus={() => setShowDestinationResults(true)}
                                    className="flex-1"
                                />
                            </div>
                            {showDestinationResults && destinationResults.length > 0 && (
                                <div className="absolute z-20 w-full mt-1 bg-white border rounded-lg shadow-lg max-h-60 overflow-y-auto">
                                    {destinationResults.map((result, idx) => (
                                        <button
                                            key={idx}
                                            className="w-full text-left p-3 hover:bg-gray-50 border-b last:border-b-0 transition-colors"
                                            onClick={() => handleDestinationSelect(result.lat, result.lon, result.display_name)}
                                        >
                                            <div className="font-medium text-sm">{result.display_name.split(',')[0]}</div>
                                            <div className="text-xs text-gray-500">{result.display_name}</div>
                                        </button>
                                    ))}
                                </div>
                            )}
                        </div>

                        {/* Transport Mode */}
                        <div className="space-y-2">
                            <label className="text-sm font-medium">Transport Mode</label>
                            <Select value={mode} onValueChange={(value) => setMode(value as TransportMode)}>
                                <SelectTrigger>
                                    <SelectValue />
                                </SelectTrigger>
                                <SelectContent>
                                    <SelectItem value="driving">Driving</SelectItem>
                                    <SelectItem value="walking">Walking</SelectItem>
                                    <SelectItem value="metro">Metro</SelectItem>
                                    <SelectItem value="combined">Combined</SelectItem>
                                </SelectContent>
                            </Select>
                        </div>

                        {/* Calculate Button */}
                        <Button
                            className="w-full"
                            onClick={handleCalculateRoute}
                            disabled={!origin || !destination || isPending}
                        >
                            {isPending ? (
                                <>
                                    <div className="mr-2 h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent" />
                                    Calculating Routes...
                                </>
                            ) : (
                                <>
                                    <Navigation className="mr-2 h-4 w-4" />
                                    Calculate Safe Routes
                                </>
                            )}
                        </Button>
                    </CardContent>
                </Card>

                {/* Error Display */}
                {isError && (
                    <Card className="border-red-200 bg-red-50">
                        <CardContent className="pt-6">
                            <div className="flex items-center text-red-800">
                                <AlertCircle className="mr-2 h-5 w-5" />
                                <span className="font-medium">Unable to calculate route. Please try again.</span>
                            </div>
                        </CardContent>
                    </Card>
                )}

                {/* Route Options */}
                {routeData && routeData.routes && routeData.routes.length > 0 && (
                    <div className="space-y-3">
                        <h2 className="text-lg font-semibold">Available Routes</h2>
                        {routeData.routes.map((route) => (
                            <Card
                                key={route.id}
                                className={`cursor-pointer transition-all ${
                                    selectedRoute?.id === route.id
                                        ? 'ring-2 ring-blue-600 bg-blue-50'
                                        : 'hover:bg-gray-50'
                                }`}
                                onClick={() => handleRouteSelection(route)}
                            >
                                <CardContent className="pt-6">
                                    <div className="flex items-start justify-between mb-3">
                                        <div className="flex items-center space-x-2">
                                            {getRouteIcon(route.type)}
                                            <span className="font-semibold">{getRouteLabel(route.type)}</span>
                                        </div>
                                        <Badge className={getRiskBadgeColor(route.risk_level)}>
                                            {route.risk_level.toUpperCase()} RISK
                                        </Badge>
                                    </div>

                                    <div className="grid grid-cols-3 gap-4 mb-3">
                                        <div>
                                            <div className="text-sm text-gray-600">Distance</div>
                                            <div className="font-semibold">{formatDistance(route.distance_meters)}</div>
                                        </div>
                                        <div>
                                            <div className="text-sm text-gray-600">Safety Score</div>
                                            <div className="font-semibold text-green-600">{route.safety_score}/100</div>
                                        </div>
                                        <div>
                                            <div className="text-sm text-gray-600">Flood Zones</div>
                                            <div className="font-semibold text-orange-600">{route.flood_intersections}</div>
                                        </div>
                                    </div>

                                    {route.flood_intersections > 0 && (
                                        <div className="flex items-center text-sm text-orange-700 bg-orange-50 p-2 rounded">
                                            <AlertCircle className="mr-2 h-4 w-4" />
                                            This route crosses {route.flood_intersections} flood-affected area{route.flood_intersections > 1 ? 's' : ''}
                                        </div>
                                    )}
                                </CardContent>
                            </Card>
                        ))}
                    </div>
                )}

                {/* No Routes Message */}
                {routeData && routeData.routes && routeData.routes.length === 0 && (
                    <Card className="border-yellow-200 bg-yellow-50">
                        <CardContent className="pt-6">
                            <div className="flex items-center text-yellow-800">
                                <AlertCircle className="mr-2 h-5 w-5" />
                                <span className="font-medium">No routes found between these locations.</span>
                            </div>
                        </CardContent>
                    </Card>
                )}
            </div>
        </div>
    );
}
