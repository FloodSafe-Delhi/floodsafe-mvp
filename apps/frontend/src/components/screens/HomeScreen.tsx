import { useState, useEffect } from 'react';
import { Card } from '../ui/card';
import { Badge } from '../ui/badge';
import {
    MapPin, Users, AlertTriangle, Bell, Shield, Phone, Camera,
    Navigation, ChevronRight, AlertCircle, Droplets,
    Maximize2, Target, RefreshCw, Info, Share2, ThumbsUp, TrendingUp, Settings, MapPinned
} from 'lucide-react';
import { FloodAlert } from '../../types';
import MapComponent from '../MapComponent';
import { useSensors, useReports, useUsers, useActiveReporters, useNearbyReporters, useLocationDetails } from '../../lib/api/hooks';
import { toast } from 'sonner';
import { cn } from '../../lib/utils';
import { getNestedValue, getNestedArray, hasLocationData } from '../../lib/safe-access';
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from '../ui/select';
import {
    Dialog,
    DialogContent,
    DialogDescription,
    DialogHeader,
    DialogTitle,
} from '../ui/dialog';

interface HomeScreenProps {
    onAlertClick: (alert: FloodAlert) => void;
    onNavigateToMap?: () => void;
    onNavigateToReport?: () => void;
    onNavigateToAlerts?: () => void;
    onNavigateToProfile?: () => void;
}

// Refresh interval options - Updated as per requirement: 15s, 2m, 10m (default)
const REFRESH_INTERVALS = {
    '15s': 15000,
    '2m': 120000,
    '10m': 600000,
} as const;

type RefreshInterval = keyof typeof REFRESH_INTERVALS;

export function HomeScreen({
    onAlertClick,
    onNavigateToMap,
    onNavigateToReport,
    onNavigateToAlerts,
    onNavigateToProfile
}: HomeScreenProps) {
    const [isRefreshing, setIsRefreshing] = useState(false);
    const [refreshInterval, setRefreshInterval] = useState<RefreshInterval>('10m'); // Default 10 minutes
    const [selectedLocation, setSelectedLocation] = useState<{ lat: number; lng: number } | null>(null);

    // User's current location (for demonstration, using Delhi coordinates)
    // In production, this would come from geolocation API
    const userLocation = { latitude: 28.6139, longitude: 77.2090 };

    const { data: sensors, refetch: refetchSensors } = useSensors();
    const { data: reports, refetch: refetchReports } = useReports();
    const { data: users } = useUsers();
    const { data: activeReportersData } = useActiveReporters();
    const { data: nearbyReportersData } = useNearbyReporters(userLocation.latitude, userLocation.longitude, 5.0);
    const { data: locationDetails } = useLocationDetails(
        selectedLocation?.lat || null,
        selectedLocation?.lng || null,
        500 // 500 meter radius
    );

    // Transform sensors into alerts with location info
    const activeAlerts: FloodAlert[] = (sensors ?? [])
        .filter(s => s.status !== 'active')
        .map(s => ({
            id: s.id,
            level: s.status === 'critical' ? 'critical' : 'warning',
            location: `Sensor ${s.id.substring(0, 8)}`,
            description: `Water level is ${s.status}.`,
            timeUntil: 'Now',
            confidence: 90,
            isActive: true,
            color: s.status === 'critical' ? 'red' : 'orange',
            coordinates: [s.longitude, s.latitude]
        }));

    // Community stats with proper logic
    const activeReporters = activeReportersData?.count || 0; // Users with reports in past 7 days
    const nearbyReporters = nearbyReportersData?.count || 0; // Users who reported within 5km
    const currentUser = users?.[0]; // Mock current user - in real app would come from auth
    const userImpact = {
        reports: currentUser?.reports_count || 0,
        helped: (currentUser?.reports_count || 0) * 15, // Rough estimate
    };

    // Determine risk level
    const riskLevel = activeAlerts.length === 0 ? 'low' :
                      activeAlerts.some(a => a.level === 'critical') ? 'severe' :
                      activeAlerts.length > 2 ? 'high' : 'moderate';

    const riskColors = {
        low: 'bg-green-500',
        moderate: 'bg-yellow-500',
        high: 'bg-orange-500',
        severe: 'bg-red-500'
    };

    const riskLabels = {
        low: 'LOW FLOOD RISK',
        moderate: 'MODERATE FLOOD RISK',
        high: 'HIGH FLOOD RISK',
        severe: 'SEVERE FLOOD RISK'
    };

    // Auto-refresh with configurable interval
    useEffect(() => {
        const intervalMs = REFRESH_INTERVALS[refreshInterval];

        const interval = setInterval(() => {
            setIsRefreshing(true);
            Promise.all([refetchSensors(), refetchReports()])
                .finally(() => {
                    setTimeout(() => setIsRefreshing(false), 1000);
                });
        }, intervalMs);

        return () => clearInterval(interval);
    }, [refreshInterval, refetchSensors, refetchReports]);

    const handleRefresh = () => {
        setIsRefreshing(true);
        Promise.all([refetchSensors(), refetchReports()])
            .finally(() => {
                setTimeout(() => setIsRefreshing(false), 1000);
                toast.success('Data refreshed successfully');
            });
    };

    const handleSOS = () => {
        toast.error('Emergency SOS activated! Contacting authorities...', {
            duration: 5000,
        });
    };

    const handleViewDetails = () => {
        if (activeAlerts.length > 0) {
            onAlertClick(activeAlerts[0]);
        } else {
            toast.info('No active alerts at this time');
        }
    };

    const handleSetAlerts = () => {
        onNavigateToAlerts?.();
        toast.success('Opening alert settings');
    };

    const handleAreaDetails = () => {
        toast.info('Viewing Whitefield area details');
    };

    const handleViewAllAlerts = () => {
        onNavigateToAlerts?.();
    };

    const handleNavigateRoutes = () => {
        onNavigateToMap?.();
        toast.success('Opening safe routes map');
    };

    const handleShare = (alertId: string) => {
        toast.success('Alert shared successfully');
    };

    const handleThankReporter = () => {
        toast.success('Thank you sent to reporter!');
    };

    const handleJoinAmbassadors = () => {
        onNavigateToProfile?.();
        toast.success('Opening ambassador program');
    };

    const handleViewLeaderboard = () => {
        onNavigateToProfile?.();
        toast.info('Viewing community leaderboard');
    };

    const handleFullscreenMap = () => {
        onNavigateToMap?.();
        toast.info('Opening full Flood Atlas');
    };

    const handleCenterMap = () => {
        toast.info('Centering map on your location');
    };

    const handleLocateAlert = (lat: number, lng: number, locationName: string) => {
        setSelectedLocation({ lat, lng });
        toast.info(`Locating ${locationName} on map`);
    };

    const formatTimeAgo = (timestamp: string) => {
        const now = new Date();
        const time = new Date(timestamp);
        const diff = now.getTime() - time.getTime();
        const minutes = Math.floor(diff / 60000);

        if (minutes < 1) return 'Just now';
        if (minutes === 1) return '1 min ago';
        if (minutes < 60) return `${minutes} min ago`;

        const hours = Math.floor(minutes / 60);
        if (hours === 1) return '1 hour ago';
        return `${hours} hours ago`;
    };

    return (
        <div className="min-h-screen bg-gradient-to-b from-white to-blue-50 pb-20 overflow-y-auto">
            {/* Dynamic Risk Header */}
            <div className={cn(riskColors[riskLevel], 'text-white p-4 shadow-lg')}>
                <div className="flex items-center justify-between flex-wrap gap-2">
                    <div className="flex-1 min-w-[200px]">
                        <div className="flex items-center gap-2 text-lg font-bold">
                            <AlertTriangle className="w-5 h-5" />
                            {riskLabels[riskLevel]}
                        </div>
                        <div className="text-sm opacity-90 mt-1">
                            Next 12 hours • Rajendra Nagar Area
                        </div>
                    </div>
                    <div className="flex gap-2">
                        <button
                            onClick={handleViewDetails}
                            className="bg-white/20 backdrop-blur px-3 py-1 rounded text-sm hover:bg-white/30 transition-colors min-h-[44px]"
                        >
                            View Details
                        </button>
                        <button
                            onClick={handleSetAlerts}
                            className="bg-white text-yellow-600 px-3 py-1 rounded text-sm font-medium hover:bg-gray-100 transition-colors min-h-[44px]"
                        >
                            Set Alerts
                        </button>
                    </div>
                </div>
            </div>

            {/* Smart Summary Cards */}
            <div className="grid grid-cols-3 gap-3 p-4">
                <Card className="p-3 border-l-4 border-blue-500 hover:shadow-lg transition-shadow">
                    <div className="flex items-center gap-2 text-gray-600 text-xs mb-1">
                        <MapPin className="w-4 h-4" />
                        <span>Your Area</span>
                    </div>
                    <div className="font-bold text-sm md:text-base">Whitefield</div>
                    <div className="text-green-600 text-xs">Low Risk</div>
                    <button
                        onClick={handleAreaDetails}
                        className="text-blue-500 text-xs mt-2 flex items-center gap-1 hover:underline min-h-[32px]"
                    >
                        Details <ChevronRight className="w-3 h-3" />
                    </button>
                </Card>

                <Card className="p-3 border-l-4 border-yellow-500 hover:shadow-lg transition-shadow">
                    <div className="flex items-center gap-2 text-gray-600 text-xs mb-1">
                        <Bell className="w-4 h-4" />
                        <span>Alerts</span>
                    </div>
                    <div className="font-bold text-sm md:text-base">{activeAlerts.length} Active</div>
                    <div className="text-gray-600 text-xs">2 hrs ahead</div>
                    <button
                        onClick={handleViewAllAlerts}
                        className="text-blue-500 text-xs mt-2 flex items-center gap-1 hover:underline min-h-[32px]"
                    >
                        View All <ChevronRight className="w-3 h-3" />
                    </button>
                </Card>

                <Card className="p-3 border-l-4 border-green-500 hover:shadow-lg transition-shadow">
                    <div className="flex items-center gap-2 text-gray-600 text-xs mb-1">
                        <Shield className="w-4 h-4" />
                        <span>Safety</span>
                    </div>
                    <div className="font-bold text-sm md:text-base">3 Routes</div>
                    <div className="text-gray-600 text-xs">Available</div>
                    <button
                        onClick={handleNavigateRoutes}
                        className="text-blue-500 text-xs mt-2 flex items-center gap-1 hover:underline min-h-[32px]"
                    >
                        Navigate <ChevronRight className="w-3 h-3" />
                    </button>
                </Card>
            </div>

            {/* Quick Action Buttons */}
            <div className="px-4 pb-3">
                <div className="grid grid-cols-3 gap-3">
                    <button
                        onClick={handleSOS}
                        className="flex flex-col items-center gap-2 p-4 bg-white rounded-lg shadow-md hover:shadow-lg transition-all border-2 border-red-100 hover:border-red-300 min-h-[100px]"
                    >
                        <div className="bg-red-500 text-white p-3 rounded-full">
                            <Phone className="w-6 h-6" />
                        </div>
                        <span className="text-sm font-semibold text-gray-700">SOS</span>
                    </button>

                    <button
                        onClick={onNavigateToReport}
                        className="flex flex-col items-center gap-2 p-4 bg-white rounded-lg shadow-md hover:shadow-lg transition-all border-2 border-blue-100 hover:border-blue-300 min-h-[100px]"
                    >
                        <div className="bg-blue-500 text-white p-3 rounded-full">
                            <Camera className="w-6 h-6" />
                        </div>
                        <span className="text-sm font-semibold text-gray-700">Report</span>
                    </button>

                    <button
                        onClick={handleNavigateRoutes}
                        className="flex flex-col items-center gap-2 p-4 bg-white rounded-lg shadow-md hover:shadow-lg transition-all border-2 border-green-100 hover:border-green-300 min-h-[100px]"
                    >
                        <div className="bg-green-500 text-white p-3 rounded-full">
                            <Navigation className="w-6 h-6" />
                        </div>
                        <span className="text-sm font-semibold text-gray-700">Routes</span>
                    </button>
                </div>
            </div>

            {/* Enhanced Map View */}
            <div className="px-4 pb-3">
                <Card className="overflow-hidden">
                    <div className="relative h-48 bg-gradient-to-br from-blue-100 to-blue-200">
                        <MapComponent className="w-full h-full" />

                        {/* Floating indicators */}
                        <div className="absolute top-4 left-4 bg-green-500 text-white px-2 py-1 rounded-full text-xs animate-pulse">
                            Sensor Active
                        </div>
                        <div className="absolute top-12 right-8 bg-yellow-500 text-white p-2 rounded-full">
                            <Droplets className="w-3 h-3" />
                        </div>
                        {activeAlerts.length > 0 && (
                            <div className="absolute bottom-8 left-12 bg-red-500 text-white p-2 rounded-full animate-pulse">
                                <AlertCircle className="w-3 h-3" />
                            </div>
                        )}

                        {/* Map controls - Vertical stack on the right */}
                        <div className="absolute bottom-2 right-2 flex flex-col gap-2">
                            <button
                                onClick={handleFullscreenMap}
                                className="bg-white p-2 rounded shadow hover:bg-gray-100 transition-colors min-h-[44px] min-w-[44px] flex items-center justify-center"
                                aria-label="Open full Flood Atlas"
                                title="Zoom / Full Map"
                            >
                                <Maximize2 className="w-4 h-4" />
                            </button>
                            <button
                                onClick={handleCenterMap}
                                className="bg-white p-2 rounded shadow hover:bg-gray-100 transition-colors min-h-[44px] min-w-[44px] flex items-center justify-center"
                                aria-label="Center on my location"
                                title="My Location"
                            >
                                <Target className="w-4 h-4" />
                            </button>
                        </div>
                    </div>
                </Card>
            </div>

            {/* Live Updates Feed with Auto-Refresh Settings */}
            <div className="px-4 pb-3">
                <Card>
                    <div className="p-3 border-b flex items-center justify-between">
                        <h3 className="font-semibold flex items-center gap-2">
                            Recent Updates
                            <RefreshCw className={cn('w-4 h-4 text-blue-500', isRefreshing && 'animate-spin')} />
                        </h3>
                        <div className="flex items-center gap-2">
                            <Select value={refreshInterval} onValueChange={(value) => setRefreshInterval(value as RefreshInterval)}>
                                <SelectTrigger className="w-[100px] h-8 text-xs">
                                    <Settings className="w-3 h-3 mr-1" />
                                    <SelectValue />
                                </SelectTrigger>
                                <SelectContent>
                                    <SelectItem value="15s">15 sec</SelectItem>
                                    <SelectItem value="2m">2 min</SelectItem>
                                    <SelectItem value="10m">10 min</SelectItem>
                                </SelectContent>
                            </Select>
                            <button
                                onClick={handleRefresh}
                                className="text-xs text-blue-500 hover:underline min-h-[32px] px-2"
                            >
                                Refresh
                            </button>
                        </div>
                    </div>

                    <div className="divide-y">
                        {/* Sensor Alerts with Location and Locate Button */}
                        {activeAlerts.length > 0 ? (
                            activeAlerts.slice(0, 2).map((alert, index) => (
                                <div key={alert.id} className="p-3">
                                    <div className="flex items-start gap-3">
                                        <div className={cn(
                                            'p-2 rounded-full flex-shrink-0',
                                            alert.level === 'critical' ? 'bg-red-100 text-red-600' : 'bg-yellow-100 text-yellow-600'
                                        )}>
                                            <AlertTriangle className="w-4 h-4" />
                                        </div>
                                        <div className="flex-1 min-w-0">
                                            <div className="text-xs text-gray-500">{index === 0 ? '2 min ago' : '15 min ago'}</div>
                                            <div className="font-medium text-sm mt-1">
                                                {alert.level === 'critical' ? 'High' : 'Moderate'} water detected - {alert.location}
                                            </div>
                                            <div className="text-sm text-gray-600">
                                                {alert.description}
                                            </div>
                                            {/* Location Display */}
                                            <div className="flex items-center gap-1 mt-1 text-xs text-gray-500">
                                                <MapPin className="w-3 h-3" />
                                                <span>
                                                    {alert.coordinates[1].toFixed(4)}, {alert.coordinates[0].toFixed(4)}
                                                </span>
                                            </div>
                                            <div className="flex gap-2 mt-2 flex-wrap">
                                                <button
                                                    onClick={() => onAlertClick(alert)}
                                                    className="text-xs bg-blue-50 text-blue-600 px-2 py-1 rounded hover:bg-blue-100 transition-colors min-h-[32px]"
                                                >
                                                    View
                                                </button>
                                                <button
                                                    onClick={() => handleLocateAlert(alert.coordinates[1], alert.coordinates[0], alert.location)}
                                                    className="text-xs bg-purple-50 text-purple-600 px-2 py-1 rounded hover:bg-purple-100 transition-colors min-h-[32px] flex items-center gap-1"
                                                >
                                                    <MapPinned className="w-3 h-3" />
                                                    Locate
                                                </button>
                                                <button
                                                    onClick={() => handleShare(alert.id)}
                                                    className="text-xs bg-gray-50 text-gray-600 px-2 py-1 rounded hover:bg-gray-100 transition-colors min-h-[32px]"
                                                >
                                                    <Share2 className="w-3 h-3 inline mr-1" />
                                                    Share
                                                </button>
                                                <button
                                                    onClick={handleNavigateRoutes}
                                                    className="text-xs bg-green-50 text-green-600 px-2 py-1 rounded hover:bg-green-100 transition-colors min-h-[32px]"
                                                >
                                                    Alt Routes
                                                </button>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            ))
                        ) : null}

                        {/* Community Reports with Location and Locate Button */}
                        {reports && reports.length > 0 ? (
                            reports.slice(0, 2).map((report) => (
                                <div key={report.id} className="p-3">
                                    <div className="flex items-start gap-3">
                                        <div className={cn(
                                            'p-2 rounded-full flex-shrink-0',
                                            report.verified ? 'bg-green-100 text-green-600' : 'bg-yellow-100 text-yellow-600'
                                        )}>
                                            <Users className="w-4 h-4" />
                                        </div>
                                        <div className="flex-1 min-w-0">
                                            <div className="text-xs text-gray-500">{formatTimeAgo(report.timestamp)}</div>
                                            <div className="font-medium text-sm mt-1">
                                                {report.verified ? 'Community Report Verified' : 'Community Report'}
                                            </div>
                                            <div className="text-sm text-gray-600 line-clamp-2">
                                                {report.description}
                                            </div>
                                            {/* Location Display */}
                                            <div className="flex items-center gap-1 mt-1 text-xs text-gray-500">
                                                <MapPin className="w-3 h-3" />
                                                <span>
                                                    {report.latitude.toFixed(4)}, {report.longitude.toFixed(4)}
                                                </span>
                                            </div>
                                            <div className="flex gap-2 mt-2 flex-wrap">
                                                <button
                                                    onClick={() => toast.info('Viewing report details')}
                                                    className="text-xs bg-blue-50 text-blue-600 px-2 py-1 rounded hover:bg-blue-100 transition-colors min-h-[32px]"
                                                >
                                                    View
                                                </button>
                                                <button
                                                    onClick={() => handleLocateAlert(report.latitude, report.longitude, 'Report Location')}
                                                    className="text-xs bg-purple-50 text-purple-600 px-2 py-1 rounded hover:bg-purple-100 transition-colors min-h-[32px] flex items-center gap-1"
                                                >
                                                    <MapPinned className="w-3 h-3" />
                                                    Locate
                                                </button>
                                                {!report.verified && (
                                                    <button
                                                        onClick={handleThankReporter}
                                                        className="text-xs bg-amber-50 text-amber-600 px-2 py-1 rounded hover:bg-amber-100 transition-colors min-h-[32px]"
                                                    >
                                                        <ThumbsUp className="w-3 h-3 inline mr-1" />
                                                        Thank
                                                    </button>
                                                )}
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            ))
                        ) : null}

                        {/* All Clear Message */}
                        {activeAlerts.length === 0 && (!reports || reports.length === 0) && (
                            <div className="p-3">
                                <div className="flex items-start gap-3">
                                    <div className="bg-green-100 text-green-600 p-2 rounded-full">
                                        <Users className="w-4 h-4" />
                                    </div>
                                    <div className="flex-1">
                                        <div className="text-xs text-gray-500">Just now</div>
                                        <div className="font-medium text-sm mt-1">
                                            All Systems Normal
                                        </div>
                                        <div className="text-sm text-gray-600">
                                            No flood alerts in your area. Stay safe!
                                        </div>
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>
                </Card>
            </div>

            {/* Community Engagement Widget - Under Recent Updates with Proper Logic */}
            <div className="px-4 pb-4">
                <div className="bg-gradient-to-r from-blue-500 to-purple-500 text-white rounded-lg shadow-md p-4">
                    <div className="flex items-center justify-between mb-3">
                        <h3 className="font-semibold flex items-center gap-2">
                            <Users className="w-5 h-5" />
                            Community Safety Network
                        </h3>
                        <Info className="w-4 h-4 opacity-70" />
                    </div>

                    <div className="grid grid-cols-2 gap-3 mb-3">
                        <div>
                            <div className="text-2xl font-bold">{activeReporters}</div>
                            <div className="text-xs opacity-90">Active Reporters</div>
                            <div className="text-[10px] opacity-75">Past 7 days</div>
                        </div>
                        <div>
                            <div className="text-2xl font-bold">{nearbyReporters}</div>
                            <div className="text-xs opacity-90">Near You</div>
                            <div className="text-[10px] opacity-75">Within 5km</div>
                        </div>
                    </div>

                    <div className="bg-white/20 backdrop-blur rounded p-2 mb-3">
                        <div className="flex items-center gap-2">
                            <TrendingUp className="w-4 h-4" />
                            <span className="text-sm">Your Impact: {userImpact.reports} reports, {userImpact.helped} people helped</span>
                        </div>
                    </div>

                    <div className="flex gap-2">
                        <button
                            onClick={handleJoinAmbassadors}
                            className="flex-1 bg-white text-blue-600 py-2 rounded font-medium text-sm hover:bg-gray-100 transition-colors min-h-[44px]"
                        >
                            Join Ambassadors
                        </button>
                        <button
                            onClick={handleViewLeaderboard}
                            className="flex-1 bg-white/20 backdrop-blur py-2 rounded text-sm hover:bg-white/30 transition-colors min-h-[44px]"
                        >
                            Leaderboard
                        </button>
                    </div>
                </div>
            </div>

            {/* Location Details Dialog */}
            <Dialog open={selectedLocation !== null} onOpenChange={(open) => !open && setSelectedLocation(null)}>
                <DialogContent className="max-w-2xl max-h-[80vh] overflow-y-auto">
                    <DialogHeader>
                        <DialogTitle>Location Details</DialogTitle>
                        <DialogDescription>
                            Reports and sensor data at this location
                        </DialogDescription>
                    </DialogHeader>

                    {locationDetails && (
                        <div className="space-y-4">
                            {hasLocationData(locationDetails.location) && (
                                <div className="text-sm text-gray-600">
                                    <div className="flex items-center gap-2">
                                        <MapPin className="w-4 h-4" />
                                        <span>
                                            {locationDetails.location.latitude.toFixed(4)}, {locationDetails.location.longitude.toFixed(4)}
                                        </span>
                                    </div>
                                    <div className="mt-1">
                                        Search Radius: {locationDetails.location.radius_meters || 500}m
                                    </div>
                                </div>
                            )}

                            <div>
                                <h4 className="font-semibold mb-2">
                                    Total Reports: {locationDetails.total_reports || 0}
                                </h4>

                                {getNestedArray(locationDetails, ['reports']).length > 0 ? (
                                    <div className="space-y-2">
                                        {getNestedArray(locationDetails, ['reports']).map((report: any) => (
                                            <Card key={report.id} className="p-3">
                                                <div className="flex items-start justify-between">
                                                    <div className="flex-1">
                                                        <div className="text-sm font-medium">{report.description}</div>
                                                        <div className="text-xs text-gray-500 mt-1">
                                                            {formatTimeAgo(report.timestamp)}
                                                        </div>
                                                        {report.verified && (
                                                            <Badge className="mt-1 bg-green-500 text-white text-xs">Verified</Badge>
                                                        )}
                                                    </div>
                                                    <div className="text-xs text-gray-500">
                                                        {report.upvotes} upvotes
                                                    </div>
                                                </div>
                                            </Card>
                                        ))}
                                    </div>
                                ) : (
                                    <p className="text-sm text-gray-500">No reports at this location</p>
                                )}
                            </div>

                            <div>
                                <h4 className="font-semibold mb-2">
                                    Reporters ({getNestedArray(locationDetails, ['reporters']).length})
                                </h4>

                                {getNestedArray(locationDetails, ['reporters']).length > 0 ? (
                                    <div className="space-y-2">
                                        {getNestedArray(locationDetails, ['reporters']).map((reporter: any) => (
                                            <Card key={reporter.id} className="p-3">
                                                <div className="flex items-center justify-between">
                                                    <div>
                                                        <div className="font-medium text-sm">{reporter.username}</div>
                                                        <div className="text-xs text-gray-500">
                                                            Level {reporter.level} • {reporter.reports_count} total reports • {reporter.verified_reports_count} verified
                                                        </div>
                                                    </div>
                                                </div>
                                            </Card>
                                        ))}
                                    </div>
                                ) : (
                                    <p className="text-sm text-gray-500">No reporter information available</p>
                                )}
                            </div>

                            <button
                                onClick={() => {
                                    onNavigateToMap?.();
                                    setSelectedLocation(null);
                                }}
                                className="w-full bg-blue-500 text-white py-2 rounded hover:bg-blue-600 transition-colors min-h-[44px]"
                            >
                                View on Full Map
                            </button>
                        </div>
                    )}

                    {!locationDetails && selectedLocation && (
                        <div className="text-center py-8">
                            <RefreshCw className="w-8 h-8 animate-spin mx-auto mb-2 text-gray-400" />
                            <p className="text-sm text-gray-500">Loading location details...</p>
                        </div>
                    )}
                </DialogContent>
            </Dialog>
        </div>
    );
}
