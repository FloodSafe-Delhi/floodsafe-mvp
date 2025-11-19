import React, { useState } from 'react';
import { Button } from './ui/button';
import { ChevronDown, ChevronUp } from 'lucide-react';

interface MapLegendProps {
    className?: string;
}

export default function MapLegend({ className }: MapLegendProps) {
    const [isExpanded, setIsExpanded] = useState(false);

    return (
        <div className={`bg-white rounded-lg shadow-lg ${className}`}>
            {/* Header */}
            <div className="flex items-center justify-between p-3 border-b">
                <h3 className="text-sm font-semibold text-gray-900">Map Legend</h3>
                <Button
                    size="icon"
                    variant="ghost"
                    onClick={() => setIsExpanded(!isExpanded)}
                    className="h-6 w-6 p-0"
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

                    {/* Routes */}
                    <div>
                        <h4 className="text-xs font-medium text-gray-700 mb-2">Routes</h4>
                        <div className="space-y-1.5">
                            <LegendItem color="#22c55e" label="Safe Route" shape="line" />
                            <LegendItem color="#ef4444" label="Flooded Route" shape="line" />
                        </div>
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
}

function LegendItem({ color, label, shape = 'square' }: LegendItemProps) {
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
                <div className="w-6 h-0.5 rounded" style={{ backgroundColor: color }} />
            )}
            <span className="text-xs text-gray-600">{label}</span>
        </div>
    );
}
