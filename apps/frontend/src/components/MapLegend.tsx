import { useState } from 'react';
import { Button } from './ui/button';
import { ChevronDown, ChevronUp } from 'lucide-react';

interface MapLegendProps {
    className?: string;
}

export default function MapLegend({ className }: MapLegendProps) {
    const [isExpanded, setIsExpanded] = useState(false);

    return (
        <div className={`bg-white rounded-lg shadow-xl border border-gray-200 ${className}`}>
            {/* Header */}
            <div className="flex items-center justify-between p-3 border-b">
                <h3 className="text-sm font-semibold text-gray-900">Map Legend</h3>
                <Button
                    size="icon"
                    variant="ghost"
                    onClick={() => setIsExpanded(!isExpanded)}
                    className="h-11 w-11 p-0"
                >
                    {isExpanded ? (
                        <ChevronDown className="h-4 w-4" />
                    ) : (
                        <ChevronUp className="h-4 w-4" />
                    )}
                </Button>
            </div>

            {/* Legend Content */}
            {isExpanded && (
                <div className="p-3 space-y-3">
                    {/* Flood Zones */}
                    <div>
                        <h4 className="text-xs font-medium text-gray-700 mb-2">Flood Zones</h4>
                        <div className="space-y-1.5">
                            <LegendItem color="#22c55e" label="Low Risk" />
                            <LegendItem color="#eab308" label="Medium Risk" />
                            <LegendItem color="#f97316" label="High Risk" />
                            <LegendItem color="#ef4444" label="Critical" />
                        </div>
                    </div>

                    {/* Sensor Status */}
                    <div>
                        <h4 className="text-xs font-medium text-gray-700 mb-2">Sensors</h4>
                        <div className="space-y-1.5">
                            <LegendItem color="#22c55e" label="Active" shape="circle" />
                            <LegendItem color="#f97316" label="Warning" shape="circle" />
                            <LegendItem color="#ef4444" label="Critical" shape="circle" />
                        </div>
                    </div>

                    {/* Community Reports */}
                    <div>
                        <h4 className="text-xs font-medium text-gray-700 mb-2">Community Reports</h4>
                        <div className="space-y-1.5">
                            <LegendItem color="#3b82f6" label="Ankle Deep" shape="circle" />
                            <LegendItem color="#f59e0b" label="Knee Deep" shape="circle" />
                            <LegendItem color="#f97316" label="Waist Deep" shape="circle" />
                            <LegendItem color="#ef4444" label="Impassable" shape="circle" />
                        </div>
                        <div className="mt-2 pt-2 border-t border-gray-200">
                            <div className="flex items-center gap-2 text-xs text-gray-600">
                                <div className="w-4 h-4 rounded-full border-2 border-green-500 bg-gray-300"></div>
                                <span>Verified</span>
                            </div>
                        </div>
                    </div>

                    {/* Routes */}
                    <div>
                        <h4 className="text-xs font-medium text-gray-700 mb-2">Routes</h4>
                        <div className="space-y-1.5">
                            <LegendItem color="#22c55e" label="Safe Route" shape="line" />
                            <LegendItem color="#ef4444" label="Flooded Route" shape="line" thickness="thick" />
                        </div>
                    </div>

                    {/* Historical Floods */}
                    <div>
                        <h4 className="text-xs font-medium text-gray-700 mb-2">Historical Floods (1967-2023)</h4>
                        <div className="space-y-1.5">
                            <LegendItem color="#22c55e" label="Minor Event" shape="circle" />
                            <LegendItem color="#eab308" label="Moderate Event" shape="circle" />
                            <LegendItem color="#ef4444" label="Severe Event" shape="circle" />
                        </div>
                        <p className="text-[10px] text-gray-500 mt-1.5">Source: India Flood Inventory</p>
                    </div>

                    {/* Flood Hazard Index (FHI) */}
                    <div>
                        <h4 className="text-xs font-semibold text-gray-700 mb-2 flex items-center gap-1">
                            <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></span>
                            Flood Hazard Index (Live)
                        </h4>
                        <div className="space-y-1.5">
                            <div className="flex items-center gap-2">
                                <div className="w-3 h-3 rounded" style={{ backgroundColor: '#22c55e' }}></div>
                                <span className="text-xs text-gray-600">Low (0-20%)</span>
                            </div>
                            <div className="flex items-center gap-2">
                                <div className="w-3 h-3 rounded" style={{ backgroundColor: '#eab308' }}></div>
                                <span className="text-xs text-gray-600">Moderate (20-40%)</span>
                            </div>
                            <div className="flex items-center gap-2">
                                <div className="w-3 h-3 rounded" style={{ backgroundColor: '#f97316' }}></div>
                                <span className="text-xs text-gray-600">High (40-70%)</span>
                            </div>
                            <div className="flex items-center gap-2">
                                <div className="w-3 h-3 rounded" style={{ backgroundColor: '#ef4444' }}></div>
                                <span className="text-xs text-gray-600">Extreme (70-100%)</span>
                            </div>
                        </div>
                        <p className="text-xs text-gray-400 mt-1.5 italic">
                            Real-time weather conditions
                        </p>
                    </div>
                </div>
            )}
        </div>
    );
}

interface LegendItemProps {
    color: string;
    label: string;
    shape?: 'square' | 'circle' | 'line';
    thickness?: 'normal' | 'thick';
}

function LegendItem({ color, label, shape = 'square', thickness = 'normal' }: LegendItemProps) {
    return (
        <div className="flex items-center gap-2">
            {shape === 'square' && (
                <div
                    className="w-4 h-4 rounded border border-gray-300"
                    style={{ backgroundColor: color, opacity: 0.6 }}
                />
            )}
            {shape === 'circle' && (
                <div
                    className="w-4 h-4 rounded-full border-2 border-white shadow-sm"
                    style={{ backgroundColor: color }}
                />
            )}
            {shape === 'line' && (
                <div
                    className={`w-6 rounded ${thickness === 'thick' ? 'h-1 shadow-md' : 'h-0.5'}`}
                    style={{ backgroundColor: color }}
                />
            )}
            <span className="text-xs text-gray-600">{label}</span>
        </div>
    );
}
