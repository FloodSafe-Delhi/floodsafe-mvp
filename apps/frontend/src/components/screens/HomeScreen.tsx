import { useState } from 'react';
import { Card } from '../ui/card';
import { Badge } from '../ui/badge';
import { MapPin, Users, Route, ChevronUp, ChevronDown } from 'lucide-react';
import { AlertStatusBadge } from '../AlertStatusBadge';
import { getAlertIcon, getAlertBorderColor } from '../../lib/utils';
import { FloodAlert } from '../../types';
import { Progress } from '../ui/progress';
import MapComponent from '../MapComponent';
import { useSensors } from '../../lib/api/hooks';

interface HomeScreenProps {
    onAlertClick: (alert: FloodAlert) => void;
}

export function HomeScreen({ onAlertClick }: HomeScreenProps) {
    const [isBottomSheetExpanded, setIsBottomSheetExpanded] = useState(false);
    const { data: sensors } = useSensors();

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

    const currentStatus = activeAlerts.length > 0 ? activeAlerts[0] : null;

    return (
        <div className="pb-16 md:pb-0 min-h-screen bg-gray-50 flex flex-col md:flex-row h-screen overflow-hidden">

            {/* Left Panel (Desktop Only) - Stats & Info */}
            <div className="md:w-96 md:h-full md:overflow-y-auto md:bg-white md:border-r md:z-10 flex-shrink-0">
                {/* Hero Section */}
                <div className="bg-white p-6 shadow-sm md:shadow-none">
                    <div className="flex flex-col items-center gap-4">
                        <AlertStatusBadge
                            level={currentStatus?.level || 'safe'}
                            color={currentStatus?.color || 'green'}
                            size="large"
                        />

                        <div className="text-center">
                            <h2 className="text-xl mb-1">
                                {currentStatus ? `${currentStatus.level.toUpperCase()} Active` : 'All Clear'}
                            </h2>
                            <p className="text-gray-600 text-sm">Real-time Sensor Network</p>
                            <p className="text-gray-500 text-xs mt-1">Live Updates</p>
                        </div>
                    </div>
                </div>

                {/* Quick Stats */}
                <div className="grid grid-cols-3 gap-3 p-4">
                    <Card className="p-3 bg-orange-50 border-orange-200">
                        <div className="text-center">
                            <div className="text-2xl mb-1">{activeAlerts.length}</div>
                            <div className="text-xs text-gray-600 flex items-center justify-center gap-1">
                                <MapPin className="w-3 h-3" />
                                Active
                            </div>
                        </div>
                    </Card>

                    <Card className="p-3 bg-green-50 border-green-200">
                        <div className="text-center">
                            <div className="text-2xl mb-1">{sensors?.filter(s => s.status === 'active').length || 0}</div>
                            <div className="text-xs text-gray-600 flex items-center justify-center gap-1">
                                <Route className="w-3 h-3" />
                                Safe
                            </div>
                        </div>
                    </Card>

                    <Card className="p-3 bg-blue-50 border-blue-200">
                        <div className="text-center">
                            <div className="text-2xl mb-1">{sensors?.length || 0}</div>
                            <div className="text-xs text-gray-600 flex items-center justify-center gap-1">
                                <Users className="w-3 h-3" />
                                Total
                            </div>
                        </div>
                    </Card>
                </div>

                {/* Desktop Alerts List */}
                <div className="hidden md:block px-4 pb-4">
                    <h3 className="mb-3 font-semibold">Active Alerts</h3>
                    <div className="space-y-3">
                        {activeAlerts.length === 0 ? (
                            <p className="text-gray-500 text-center py-4">No active alerts.</p>
                        ) : (
                            activeAlerts.map((alert) => (
                                <Card
                                    key={alert.id}
                                    className={`p-4 border-l-4 ${getAlertBorderColor(alert.color)} cursor-pointer hover:shadow-md transition-shadow`}
                                    onClick={() => onAlertClick(alert)}
                                >
                                    <div className="flex items-start gap-3">
                                        <span className="text-2xl">{getAlertIcon(alert.level)}</span>
                                        <div className="flex-1 min-w-0">
                                            <h4 className="text-sm font-medium">{alert.location}</h4>
                                            <p className="text-xs text-gray-600 mb-1">{alert.description}</p>
                                        </div>
                                    </div>
                                </Card>
                            ))
                        )}
                    </div>
                </div>
            </div>

            {/* Map Section (Full Screen on Desktop, Larger on Mobile) */}
            <div className="flex-1 relative min-h-[500px] h-[60vh] md:h-full">
                <MapComponent className="w-full h-full" />

                {/* Offline Indicator */}
                <div className="absolute top-4 right-4 pointer-events-none z-10">
                    <Badge variant="secondary" className="bg-white shadow">
                        Online
                    </Badge>
                </div>
            </div>

            {/* Mobile Bottom Sheet (Hidden on Desktop) */}
            <div className="md:hidden fixed bottom-16 left-0 right-0 bg-white shadow-2xl rounded-t-3xl transition-all duration-300 z-40"
                style={{
                    maxHeight: isBottomSheetExpanded ? '60vh' : '200px',
                    overflowY: 'auto'
                }}
            >
                {/* Handle */}
                <button
                    onClick={() => setIsBottomSheetExpanded(!isBottomSheetExpanded)}
                    className="w-full py-3 flex flex-col items-center gap-1 min-h-[44px]"
                    aria-label={isBottomSheetExpanded ? 'Collapse alerts' : 'Expand alerts'}
                >
                    <div className="w-12 h-1 bg-gray-300 rounded-full"></div>
                    {isBottomSheetExpanded ? <ChevronDown className="w-5 h-5" /> : <ChevronUp className="w-5 h-5" />}
                </button>

                <div className="px-4 pb-4">
                    <h3 className="mb-3">
                        Active Alerts ({activeAlerts.length})
                    </h3>
                    {/* Mobile Alerts List (Same as desktop but inside sheet) */}
                    <div className="space-y-3">
                        {activeAlerts.length === 0 ? (
                            <p className="text-gray-500 text-center py-4">No active alerts.</p>
                        ) : (
                            activeAlerts.map((alert) => (
                                <Card
                                    key={alert.id}
                                    className={`p-4 border-l-4 ${getAlertBorderColor(alert.color)} cursor-pointer hover:shadow-md transition-shadow`}
                                    onClick={() => onAlertClick(alert)}
                                >
                                    <div className="flex items-start gap-3">
                                        <span className="text-2xl">{getAlertIcon(alert.level)}</span>
                                        <div className="flex-1 min-w-0">
                                            <h4 className="text-sm">{alert.location}</h4>
                                            <p className="text-xs text-gray-600 mb-2">{alert.description}</p>
                                        </div>
                                    </div>
                                </Card>
                            ))
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
}
