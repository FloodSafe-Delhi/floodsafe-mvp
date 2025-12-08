import { useState } from 'react';
import MapComponent from '../MapComponent';
import { Button } from '../ui/button';
import { Input } from '../ui/input';
import { Card, CardContent } from '../ui/card';
import { Badge } from '../ui/badge';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../ui/select';
import { Navigation, MapPin, X, AlertCircle, Shield, Zap, TrendingUp } from 'lucide-react';
import { useRouteCalculation, useGeocode } from '../../lib/api/hooks';
import { useCurrentCity } from '../../contexts/CityContext';
import type { LocationPoint, RouteOption, TransportMode } from '../../types';
import { toast } from 'sonner';

export function FloodAtlasScreen() {
    const city = useCurrentCity();
    const [showRoutePanel, setShowRoutePanel] = useState(false);
    const [originQuery, setOriginQuery] = useState('');
    const [destinationQuery, setDestinationQuery] = useState('');
    const [origin, setOrigin] = useState<LocationPoint | null>(null);
    const [destination, setDestination] = useState<LocationPoint | null>(null);
    const [mode, setMode] = useState<TransportMode>('driving');
    const [activeRoute, setActiveRoute] = useState<RouteOption | null>(null);
    const [showOriginResults, setShowOriginResults] = useState(false);
    const [showDestinationResults, setShowDestinationResults] = useState(false);

    const { mutate: calculateRoute, data: routeData, isPending, isError } = useRouteCalculation();
    const { data: originResults = [] } = useGeocode(originQuery, showOriginResults && originQuery.length >= 3);
    const { data: destinationResults = [] } = useGeocode(destinationQuery, showDestinationResults && destinationQuery.length >= 3);

    const handleOriginSelect = (lat: string, lon: string, displayName: string) => {
        setOrigin({ lat: parseFloat(lat), lng: parseFloat(lon) });
        setOriginQuery(displayName.split(',')[0]);
        setShowOriginResults(false);
    };

    const handleDestinationSelect = (lat: string, lon: string, displayName: string) => {
        setDestination({ lat: parseFloat(lat), lng: parseFloat(lon) });
        setDestinationQuery(displayName.split(',')[0]);
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
                    setActiveRoute(data.routes[0]);
                } else {
                    toast.error('No routes found');
                }
            },
            onError: (error) => {
                toast.error(`Route calculation failed: ${error.message}`);
            }
        });
    };

    const formatDistance = (meters: number): string => {
        if (meters < 1000) return `${Math.round(meters)}m`;
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

    return (
        <div className="fixed inset-0 top-14 md:top-0 bottom-16 bg-transparent">
            <MapComponent
                className="w-full h-full"
                title="Flood Atlas"
                showControls={true}
                showCitySelector={true}
            />

            {/* Floating Route Button */}
            {!showRoutePanel && (
                <Button
                    onClick={() => setShowRoutePanel(true)}
                    className="fixed bottom-20 right-4 z-[200] shadow-lg"
                    size="lg"
                >
                    <Navigation className="mr-2 h-5 w-5" />
                    Plan Safe Route
                </Button>
            )}

            {/* Route Planning Panel */}
            {showRoutePanel && (
                <div className="fixed bottom-16 left-0 right-0 z-[200] max-h-[70vh] overflow-y-auto bg-white border-t shadow-2xl">
                    <div className="p-4 space-y-3">
                        {/* Header */}
                        <div className="flex items-center justify-between">
                            <h2 className="text-lg font-bold">Plan Safe Route</h2>
                            <Button
                                variant="ghost"
                                size="icon"
                                onClick={() => {
                                    setShowRoutePanel(false);
                                    setActiveRoute(null);
                                }}
                            >
                                <X className="h-5 w-5" />
                            </Button>
                        </div>

                        {/* Origin Input */}
                        <div className="relative">
                            <div className="flex items-center space-x-2">
                                <MapPin className="w-4 h-4 text-blue-600 flex-shrink-0" />
                                <Input
                                    placeholder="Starting location..."
                                    value={originQuery}
                                    onChange={(e) => {
                                        setOriginQuery(e.target.value);
                                        setShowOriginResults(true);
                                    }}
                                    className="text-sm"
                                />
                            </div>
                            {showOriginResults && originResults.length > 0 && (
                                <div className="absolute z-50 w-full mt-1 bg-white border rounded-lg shadow-lg max-h-40 overflow-y-auto">
                                    {originResults.map((result, idx) => (
                                        <button
                                            key={idx}
                                            className="w-full text-left p-2 hover:bg-gray-50 border-b last:border-b-0 text-xs"
                                            onClick={() => handleOriginSelect(result.lat, result.lon, result.display_name)}
                                        >
                                            {result.display_name}
                                        </button>
                                    ))}
                                </div>
                            )}
                        </div>

                        {/* Destination Input */}
                        <div className="relative">
                            <div className="flex items-center space-x-2">
                                <MapPin className="w-4 h-4 text-red-600 flex-shrink-0" />
                                <Input
                                    placeholder="Destination..."
                                    value={destinationQuery}
                                    onChange={(e) => {
                                        setDestinationQuery(e.target.value);
                                        setShowDestinationResults(true);
                                    }}
                                    className="text-sm"
                                />
                            </div>
                            {showDestinationResults && destinationResults.length > 0 && (
                                <div className="absolute z-50 w-full mt-1 bg-white border rounded-lg shadow-lg max-h-40 overflow-y-auto">
                                    {destinationResults.map((result, idx) => (
                                        <button
                                            key={idx}
                                            className="w-full text-left p-2 hover:bg-gray-50 border-b last:border-b-0 text-xs"
                                            onClick={() => handleDestinationSelect(result.lat, result.lon, result.display_name)}
                                        >
                                            {result.display_name}
                                        </button>
                                    ))}
                                </div>
                            )}
                        </div>

                        {/* Mode Selector */}
                        <Select value={mode} onValueChange={(value) => setMode(value as TransportMode)}>
                            <SelectTrigger className="text-sm">
                                <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                                <SelectItem value="driving">Driving</SelectItem>
                                <SelectItem value="walking">Walking</SelectItem>
                                <SelectItem value="metro">Metro</SelectItem>
                            </SelectContent>
                        </Select>

                        {/* Calculate Button */}
                        <Button
                            className="w-full"
                            onClick={handleCalculateRoute}
                            disabled={!origin || !destination || isPending}
                            size="sm"
                        >
                            {isPending ? 'Calculating...' : 'Calculate Routes'}
                        </Button>

                        {/* Route Options */}
                        {routeData && routeData.routes && routeData.routes.length > 0 && (
                            <div className="space-y-2">
                                <h3 className="text-sm font-semibold">Available Routes</h3>
                                {routeData.routes.map((route) => (
                                    <Card
                                        key={route.id}
                                        className={`cursor-pointer transition-all ${
                                            activeRoute?.id === route.id ? 'ring-2 ring-blue-600' : ''
                                        }`}
                                        onClick={() => setActiveRoute(route)}
                                    >
                                        <CardContent className="p-3">
                                            <div className="flex items-center justify-between mb-2">
                                                <div className="flex items-center space-x-1 text-xs">
                                                    {getRouteIcon(route.type)}
                                                    <span className="font-semibold capitalize">{route.type}</span>
                                                </div>
                                                <Badge className={getRiskBadgeColor(route.risk_level)}>
                                                    {route.risk_level}
                                                </Badge>
                                            </div>
                                            <div className="grid grid-cols-3 gap-2 text-xs">
                                                <div>
                                                    <div className="text-gray-600">Distance</div>
                                                    <div className="font-semibold">{formatDistance(route.distance_meters)}</div>
                                                </div>
                                                <div>
                                                    <div className="text-gray-600">Safety</div>
                                                    <div className="font-semibold text-green-600">{route.safety_score}/100</div>
                                                </div>
                                                <div>
                                                    <div className="text-gray-600">Floods</div>
                                                    <div className="font-semibold text-orange-600">{route.flood_intersections}</div>
                                                </div>
                                            </div>
                                        </CardContent>
                                    </Card>
                                ))}
                            </div>
                        )}

                        {/* Error */}
                        {isError && (
                            <div className="flex items-center text-red-700 bg-red-50 p-2 rounded text-xs">
                                <AlertCircle className="mr-2 h-4 w-4" />
                                <span>Unable to calculate route</span>
                            </div>
                        )}
                    </div>
                </div>
            )}
        </div>
    );
}
