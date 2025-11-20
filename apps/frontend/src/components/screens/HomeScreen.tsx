import { useState, useEffect } from 'react';
import { Card } from '../ui/card';
import { Badge } from '../ui/badge';
import {
    MapPin, Users, AlertTriangle, Bell, Shield, Phone, Camera,
    Navigation, Map as MapIcon, ChevronRight, AlertCircle, Droplets,
    Maximize2, Target, RefreshCw, Info, Share2, ThumbsUp, TrendingUp
} from 'lucide-react';
import { FloodAlert } from '../../types';
import MapComponent from '../MapComponent';
import { useSensors } from '../../lib/api/hooks';
import { toast } from 'sonner';
import { cn } from '../../lib/utils';

interface HomeScreenProps {
    onAlertClick: (alert: FloodAlert) => void;
    onNavigateToMap?: () => void;
    onNavigateToReport?: () => void;
    onNavigateToAlerts?: () => void;
    onNavigateToProfile?: () => void;
}

export function HomeScreen({
    onAlertClick,
    onNavigateToMap,
    onNavigateToReport,
    onNavigateToAlerts,
    onNavigateToProfile
}: HomeScreenProps) {
    const [isRefreshing, setIsRefreshing] = useState(false);
    const { data: sensors, refetch } = useSensors();

    const activeAlerts: FloodAlert[] = sensors?.filter(s => s.status !== 'active').map(s => ({
        id: s.id,
        level: s.status === 'critical' ? 'critical' : 'warning',
        location: `Sensor ${s.id.substring(0, 8)}`,
        description: `Water level is ${s.status}.`,
        timeUntil: 'Now',
        confidence: 90,
        isActive: true,
        color: s.status === 'critical' ? 'red' : 'orange',
        coordinates: [s.longitude, s.latitude]
    })) || [];

    const safeSensors = sensors?.filter(s => s.status === 'active').length || 0;
    const totalSensors = sensors?.length || 0;

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

    // Auto-refresh simulation
    useEffect(() => {
        const interval = setInterval(() => {
            setIsRefreshing(true);
            refetch().finally(() => {
                setTimeout(() => setIsRefreshing(false), 1000);
            });
        }, 30000); // Refresh every 30 seconds

        return () => clearInterval(interval);
    }, [refetch]);

    const handleRefresh = () => {
        setIsRefreshing(true);
        refetch().finally(() => {
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
    };

    const handleCenterMap = () => {
        toast.info('Centering map on your location');
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
                <Card className="p-3 flex justify-around">
                    <button
                        onClick={handleSOS}
                        className="flex flex-col items-center gap-1 px-4 py-2 rounded-lg hover:bg-red-50 transition-colors min-h-[44px] min-w-[44px]"
                    >
                        <div className="bg-red-500 text-white p-3 rounded-full">
                            <Phone className="w-5 h-5" />
                        </div>
                        <span className="text-xs font-medium">SOS</span>
                    </button>

                    <button
                        onClick={onNavigateToReport}
                        className="flex flex-col items-center gap-1 px-4 py-2 rounded-lg hover:bg-blue-50 transition-colors min-h-[44px] min-w-[44px]"
                    >
                        <div className="bg-blue-500 text-white p-3 rounded-full">
                            <Camera className="w-5 h-5" />
                        </div>
                        <span className="text-xs font-medium">Report</span>
                    </button>

                    <button
                        onClick={handleNavigateRoutes}
                        className="flex flex-col items-center gap-1 px-4 py-2 rounded-lg hover:bg-green-50 transition-colors min-h-[44px] min-w-[44px]"
                    >
                        <div className="bg-green-500 text-white p-3 rounded-full">
                            <Navigation className="w-5 h-5" />
                        </div>
                        <span className="text-xs font-medium">Routes</span>
                    </button>
                </Card>
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

                        {/* Map controls */}
                        <div className="absolute bottom-2 right-2 flex gap-2">
                            <button
                                onClick={handleFullscreenMap}
                                className="bg-white p-2 rounded shadow hover:bg-gray-100 transition-colors min-h-[44px] min-w-[44px]"
                                aria-label="Fullscreen map"
                            >
                                <Maximize2 className="w-4 h-4" />
                            </button>
                            <button
                                onClick={handleCenterMap}
                                className="bg-white p-2 rounded shadow hover:bg-gray-100 transition-colors min-h-[44px] min-w-[44px]"
                                aria-label="Center on location"
                            >
                                <Target className="w-4 h-4" />
                            </button>
                        </div>
                    </div>
                </Card>
            </div>

            {/* Live Updates Feed */}
            <div className="px-4 pb-3">
                <Card>
                    <div className="p-3 border-b flex items-center justify-between">
                        <h3 className="font-semibold flex items-center gap-2">
                            Recent Updates
                            <RefreshCw className={cn('w-4 h-4 text-blue-500', isRefreshing && 'animate-spin')} />
                        </h3>
                        <button
                            onClick={handleRefresh}
                            className="text-xs text-blue-500 hover:underline min-h-[32px]"
                        >
                            Refresh
                        </button>
                    </div>

                    <div className="divide-y">
                        {activeAlerts.length > 0 ? (
                            activeAlerts.slice(0, 2).map((alert, index) => (
                                <div key={alert.id} className="p-3">
                                    <div className="flex items-start gap-3">
                                        <div className={cn(
                                            'p-2 rounded-full',
                                            alert.level === 'critical' ? 'bg-red-100 text-red-600' : 'bg-yellow-100 text-yellow-600'
                                        )}>
                                            <AlertTriangle className="w-4 h-4" />
                                        </div>
                                        <div className="flex-1">
                                            <div className="text-xs text-gray-500">{index === 0 ? '2 min ago' : '15 min ago'}</div>
                                            <div className="font-medium text-sm mt-1">
                                                {alert.level === 'critical' ? 'High' : 'Moderate'} water detected - {alert.location}
                                            </div>
                                            <div className="text-sm text-gray-600">
                                                {alert.description}
                                            </div>
                                            <div className="flex gap-2 mt-2 flex-wrap">
                                                <button
                                                    onClick={() => onAlertClick(alert)}
                                                    className="text-xs bg-blue-50 text-blue-600 px-2 py-1 rounded hover:bg-blue-100 transition-colors min-h-[32px]"
                                                >
                                                    View
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
                        ) : (
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

            {/* Community Engagement Widget */}
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
                            <div className="text-2xl font-bold">247</div>
                            <div className="text-xs opacity-90">Active Reporters</div>
                        </div>
                        <div>
                            <div className="text-2xl font-bold">12</div>
                            <div className="text-xs opacity-90">Near You</div>
                        </div>
                    </div>

                    <div className="bg-white/20 backdrop-blur rounded p-2 mb-3">
                        <div className="flex items-center gap-2">
                            <TrendingUp className="w-4 h-4" />
                            <span className="text-sm">Your Impact: 3 reports, 45 people helped</span>
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
        </div>
    );
}
