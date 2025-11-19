import { useState } from 'react';
import { Card } from '../ui/card';
import { Badge } from '../ui/badge';
import { MapPin, Users, Route, ChevronUp, ChevronDown } from 'lucide-react';
import { AlertStatusBadge } from '../AlertStatusBadge';
import { mockAlerts } from '../../lib/mockData';
import { getAlertIcon, getAlertBorderColor } from '../../lib/utils';
import { FloodAlert } from '../../types';
import { Progress } from '../ui/progress';

interface HomeScreenProps {
  onAlertClick: (alert: FloodAlert) => void;
}

export function HomeScreen({ onAlertClick }: HomeScreenProps) {
  const [isBottomSheetExpanded, setIsBottomSheetExpanded] = useState(false);
  const activeAlerts = mockAlerts.filter(a => a.isActive);
  const currentStatus = activeAlerts.length > 0 ? activeAlerts[0] : null;

  return (
    <div className="pb-16 min-h-screen bg-gray-50">
      {/* Hero Section */}
      <div className="bg-white p-6 shadow-sm">
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
            <p className="text-gray-600 text-sm">Next update in 23 minutes</p>
            <p className="text-gray-500 text-xs mt-1">Last updated: 5:30 PM, Oct 11, 2025</p>
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
              Active Alerts
            </div>
          </div>
        </Card>

        <Card className="p-3 bg-green-50 border-green-200">
          <div className="text-center">
            <div className="text-2xl mb-1">18</div>
            <div className="text-xs text-gray-600 flex items-center justify-center gap-1">
              <Route className="w-3 h-3" />
              Safe Routes
            </div>
          </div>
        </Card>

        <Card className="p-3 bg-blue-50 border-blue-200">
          <div className="text-center">
            <div className="text-2xl mb-1">5</div>
            <div className="text-xs text-gray-600 flex items-center justify-center gap-1">
              <Users className="w-3 h-3" />
              Reports/hour
            </div>
          </div>
        </Card>
      </div>

      {/* Map Section */}
      <div className="px-4 pb-4">
        <Card className="overflow-hidden h-[300px] relative">
          <div className="w-full h-full bg-gradient-to-br from-green-100 to-blue-100 flex items-center justify-center">
            <div className="text-center text-gray-600">
              <MapPin className="w-12 h-12 mx-auto mb-2" />
              <p className="text-sm">Interactive Map</p>
              <p className="text-xs">71 hotspots monitored</p>
            </div>
          </div>
          
          {/* Map Controls */}
          <div className="absolute bottom-4 right-4 flex flex-col gap-2">
            <button className="bg-white shadow-lg rounded-full w-10 h-10 flex items-center justify-center min-w-[44px] min-h-[44px]" aria-label="Zoom in">
              +
            </button>
            <button className="bg-white shadow-lg rounded-full w-10 h-10 flex items-center justify-center min-w-[44px] min-h-[44px]" aria-label="Zoom out">
              −
            </button>
            <button className="bg-white shadow-lg rounded-full w-10 h-10 flex items-center justify-center min-w-[44px] min-h-[44px]" aria-label="My location">
              <MapPin className="w-5 h-5 text-blue-600" />
            </button>
          </div>

          {/* Offline Indicator */}
          <div className="absolute top-4 right-4">
            <Badge variant="secondary" className="bg-white shadow">
              Online
            </Badge>
          </div>
        </Card>
      </div>

      {/* Bottom Sheet */}
      <div className="fixed bottom-16 left-0 right-0 bg-white shadow-2xl rounded-t-3xl transition-all duration-300"
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
            Active Alerts in Your Area ({activeAlerts.length})
          </h3>

          <div className="space-y-3">
            {activeAlerts.map((alert) => (
              <Card 
                key={alert.id} 
                className={`p-4 border-l-4 ${getAlertBorderColor(alert.color)} cursor-pointer hover:shadow-md transition-shadow`}
                onClick={() => onAlertClick(alert)}
              >
                <div className="flex items-start gap-3">
                  <span className="text-2xl">{getAlertIcon(alert.level)}</span>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-start justify-between gap-2 mb-1">
                      <h4 className="text-sm">{alert.location}</h4>
                      <Badge variant="secondary" className="text-xs">
                        {alert.timeUntil}
                      </Badge>
                    </div>
                    <p className="text-xs text-gray-600 mb-2">{alert.description}</p>
                    <div className="space-y-1">
                      <div className="flex items-center gap-2">
                        <span className="text-xs text-gray-500">Confidence:</span>
                        <Progress value={alert.confidence} className="h-2 flex-1" />
                        <span className="text-xs">{alert.confidence}%</span>
                      </div>
                    </div>
                    <button 
                      className="text-blue-600 text-sm mt-2 hover:underline"
                      onClick={(e) => {
                        e.stopPropagation();
                        onAlertClick(alert);
                      }}
                    >
                      View Details →
                    </button>
                  </div>
                </div>
              </Card>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
