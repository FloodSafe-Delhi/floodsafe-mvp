/**
 * GPS Test Panel for Navigation Testing
 *
 * Compact floating panel for testing GPS-dependent navigation features
 * without physical movement. Simulates GPS position updates along Delhi routes.
 *
 * Only visible when VITE_ENABLE_GPS_TESTING=true
 */

import { useState, useEffect, useCallback } from 'react';
import { Button } from '../ui/button';
import { Badge } from '../ui/badge';
import { Progress } from '../ui/progress';
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from '../ui/select';
import {
    Play,
    Pause,
    Square,
    ChevronDown,
    ChevronUp,
    AlertTriangle,
    Satellite,
    RotateCcw,
} from 'lucide-react';

import { useGPSSimulator, type GPSPosition } from '../../lib/testing/useGPSSimulator';
import {
    DELHI_TEST_ROUTES,
    getAllTestRoutes,
} from '../../lib/testing/testRoutes';
import {
    installMockGeolocation,
    setMockPosition,
    isMockGeolocationInstalled,
} from '../../lib/testing/mockGeolocation';

interface GPSTestPanelProps {
    onPositionChange?: (position: GPSPosition) => void;
    defaultExpanded?: boolean;
}

export function GPSTestPanel({
    onPositionChange,
    defaultExpanded = false, // Start collapsed to reduce clutter
}: GPSTestPanelProps) {
    const [isExpanded, setIsExpanded] = useState(defaultExpanded);
    const [selectedRouteId, setSelectedRouteId] = useState<string>('cp-india-gate');
    const [isMockInstalled, setIsMockInstalled] = useState(false);
    const [speedKmh, setSpeedKmh] = useState(30);

    const selectedRoute = getAllTestRoutes().find(r => r.id === selectedRouteId)
        ?? DELHI_TEST_ROUTES.CP_TO_INDIA_GATE;

    const simulator = useGPSSimulator({
        route: selectedRoute.waypoints,
        speedKmh,
        intervalMs: 1000,
        onPositionUpdate: useCallback((position: GPSPosition) => {
            setMockPosition({
                lat: position.lat,
                lng: position.lng,
                accuracy: position.accuracy,
                speed: position.speed,
                heading: position.heading,
            });
            onPositionChange?.(position);
        }, [onPositionChange]),
        onComplete: useCallback(() => {
            console.log('[GPSTestPanel] Simulation complete');
        }, []),
    });

    // Install mock geolocation on mount
    useEffect(() => {
        if (!isMockGeolocationInstalled()) {
            installMockGeolocation();
            setIsMockInstalled(true);
            console.log('[GPSTestPanel] Mock geolocation installed');
            setMockPosition({
                lat: selectedRoute.origin.lat,
                lng: selectedRoute.origin.lng,
                accuracy: 10,
            });
        } else {
            setIsMockInstalled(true);
        }
        return () => {
            simulator.stop();
        };
    }, []);

    useEffect(() => {
        simulator.setSpeed(speedKmh);
    }, [speedKmh, simulator.setSpeed]);

    useEffect(() => {
        simulator.stop();
        setMockPosition({
            lat: selectedRoute.origin.lat,
            lng: selectedRoute.origin.lng,
            accuracy: 10,
        });
    }, [selectedRouteId]);

    const currentPos = simulator.currentPosition;

    return (
        <div
            style={{
                position: 'fixed',
                top: '80px',
                left: '16px',
                zIndex: 9999,
                width: '280px',
                pointerEvents: 'auto',
            }}
        >
            <div
                style={{
                    background: 'rgba(69, 26, 3, 0.95)',
                    backdropFilter: 'blur(8px)',
                    border: '1px solid rgba(217, 119, 6, 0.5)',
                    borderRadius: '12px',
                    boxShadow: '0 10px 25px rgba(0,0,0,0.3)',
                    overflow: 'hidden',
                }}
            >
                {/* Compact Header - Always Visible */}
                <div
                    onClick={() => setIsExpanded(!isExpanded)}
                    style={{
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        padding: '10px 12px',
                        cursor: 'pointer',
                        borderBottom: isExpanded ? '1px solid rgba(217, 119, 6, 0.3)' : 'none',
                    }}
                >
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <Satellite style={{ width: '16px', height: '16px', color: '#fbbf24' }} />
                        <span style={{ fontSize: '13px', fontWeight: 600, color: '#fef3c7' }}>
                            GPS Test
                        </span>
                        <Badge
                            variant="outline"
                            className={isMockInstalled
                                ? 'bg-green-900/50 text-green-400 border-green-600 text-xs px-1.5 py-0'
                                : 'bg-red-900/50 text-red-400 border-red-600 text-xs px-1.5 py-0'
                            }
                        >
                            {isMockInstalled ? 'MOCK' : 'REAL'}
                        </Badge>
                        {simulator.isRunning && (
                            <Badge variant="outline" className="bg-blue-900/50 text-blue-400 border-blue-600 text-xs px-1.5 py-0">
                                {simulator.progress.toFixed(0)}%
                            </Badge>
                        )}
                    </div>
                    {isExpanded
                        ? <ChevronUp style={{ width: '16px', height: '16px', color: '#fbbf24' }} />
                        : <ChevronDown style={{ width: '16px', height: '16px', color: '#fbbf24' }} />
                    }
                </div>

                {/* Expanded Content */}
                {isExpanded && (
                    <div style={{ padding: '12px', display: 'flex', flexDirection: 'column', gap: '12px' }}>
                        {/* Route Selection */}
                        <div>
                            <label style={{ fontSize: '11px', color: 'rgba(253, 230, 138, 0.7)', marginBottom: '4px', display: 'block' }}>
                                Route
                            </label>
                            <Select value={selectedRouteId} onValueChange={setSelectedRouteId}>
                                <SelectTrigger className="bg-amber-900/50 border-amber-700 text-amber-100 h-8 text-xs">
                                    <SelectValue placeholder="Select route" />
                                </SelectTrigger>
                                <SelectContent className="bg-amber-950 border-amber-700">
                                    {getAllTestRoutes().map(route => (
                                        <SelectItem
                                            key={route.id}
                                            value={route.id}
                                            className="text-amber-100 focus:bg-amber-800 focus:text-amber-50 text-xs"
                                        >
                                            {route.name}
                                        </SelectItem>
                                    ))}
                                </SelectContent>
                            </Select>
                        </div>

                        {/* Speed Control */}
                        <div>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
                                <label style={{ fontSize: '11px', color: 'rgba(253, 230, 138, 0.7)' }}>Speed</label>
                                <span style={{ fontSize: '11px', color: '#fbbf24', fontFamily: 'monospace' }}>{speedKmh} km/h</span>
                            </div>
                            <input
                                type="range"
                                min="5"
                                max="60"
                                step="5"
                                value={speedKmh}
                                onChange={(e) => setSpeedKmh(Number(e.target.value))}
                                style={{
                                    width: '100%',
                                    height: '6px',
                                    borderRadius: '3px',
                                    background: 'rgba(120, 53, 15, 0.8)',
                                    outline: 'none',
                                    cursor: 'pointer',
                                }}
                            />
                        </div>

                        {/* Controls */}
                        <div style={{ display: 'flex', gap: '6px' }}>
                            {!simulator.isRunning ? (
                                <Button onClick={() => simulator.start()} size="sm" className="flex-1 bg-green-600 hover:bg-green-700 text-white h-8 text-xs">
                                    <Play className="h-3 w-3 mr-1" /> Start
                                </Button>
                            ) : simulator.isPaused ? (
                                <Button onClick={() => simulator.resume()} size="sm" className="flex-1 bg-green-600 hover:bg-green-700 text-white h-8 text-xs">
                                    <Play className="h-3 w-3 mr-1" /> Resume
                                </Button>
                            ) : (
                                <Button onClick={() => simulator.pause()} size="sm" className="flex-1 bg-amber-600 hover:bg-amber-700 text-white h-8 text-xs">
                                    <Pause className="h-3 w-3 mr-1" /> Pause
                                </Button>
                            )}
                            <Button
                                onClick={() => simulator.stop()}
                                size="sm"
                                variant="outline"
                                className="border-red-600 text-red-400 hover:bg-red-900/50 h-8 px-2"
                                disabled={!simulator.isRunning && !simulator.isPaused}
                            >
                                <Square className="h-3 w-3" />
                            </Button>
                            <Button
                                onClick={() => {
                                    simulator.stop();
                                    setMockPosition({
                                        lat: selectedRoute.origin.lat,
                                        lng: selectedRoute.origin.lng,
                                        accuracy: 10,
                                    });
                                }}
                                size="sm"
                                variant="outline"
                                className="border-amber-600 text-amber-400 hover:bg-amber-900/50 h-8 px-2"
                            >
                                <RotateCcw className="h-3 w-3" />
                            </Button>
                        </div>

                        {/* Progress */}
                        {simulator.isRunning && (
                            <div>
                                <Progress value={simulator.progress} className="h-1.5 bg-amber-900" />
                                <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '4px' }}>
                                    <span style={{ fontSize: '10px', color: 'rgba(251, 191, 36, 0.6)' }}>
                                        Pt {simulator.currentIndex + 1}/{selectedRoute.waypoints.length}
                                    </span>
                                    <span style={{ fontSize: '10px', color: 'rgba(251, 191, 36, 0.6)' }}>
                                        {simulator.progress.toFixed(0)}%
                                    </span>
                                </div>
                            </div>
                        )}

                        {/* Deviation Test */}
                        <Button
                            onClick={() => simulator.triggerDeviation(100)}
                            size="sm"
                            variant="outline"
                            className={`w-full h-8 text-xs ${simulator.isDeviated
                                ? 'border-red-500 text-red-400 bg-red-900/30'
                                : 'border-amber-600 text-amber-400 hover:bg-amber-900/50'
                            }`}
                            disabled={!simulator.isRunning}
                        >
                            <AlertTriangle className="h-3 w-3 mr-1" />
                            {simulator.isDeviated ? 'Deviated! (100m off)' : 'Trigger Off-Route'}
                        </Button>

                        {/* Current Position - Compact */}
                        {currentPos && (
                            <div style={{
                                background: 'rgba(120, 53, 15, 0.5)',
                                borderRadius: '6px',
                                padding: '8px',
                                fontSize: '10px',
                                fontFamily: 'monospace',
                                color: '#fef3c7',
                            }}>
                                <span style={{ color: 'rgba(253, 230, 138, 0.6)' }}>Pos: </span>
                                {currentPos.lat.toFixed(5)}, {currentPos.lng.toFixed(5)}
                            </div>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
}
