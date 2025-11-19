import { useState } from 'react';
import { Settings, Filter } from 'lucide-react';
import { Card } from '../ui/card';
import { Badge } from '../ui/badge';
import { Button } from '../ui/button';
import { Progress } from '../ui/progress';
import { FloodAlert } from '../../types';
import { mockAlerts } from '../../lib/mockData';
import { getAlertIcon, getAlertBorderColor } from '../../lib/utils';

interface AlertsListScreenProps {
  onAlertClick: (alert: FloodAlert) => void;
}

export function AlertsListScreen({ onAlertClick }: AlertsListScreenProps) {
  const [filter, setFilter] = useState<'all' | 'active' | 'archived'>('all');
  const activeAlerts = mockAlerts.filter(a => a.isActive);
  const archivedAlerts = [
    {
      id: '999',
      location: 'Connaught Place',
      level: 'safe' as const,
      color: 'green' as const,
      title: 'Connaught Place - All Clear',
      resolvedAt: '3:00 PM',
      description: 'Flood warning lifted'
    }
  ];

  const displayedAlerts = filter === 'active' ? activeAlerts : 
                         filter === 'archived' ? [] : activeAlerts;

  return (
    <div className="pb-16 min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow-sm sticky top-14 z-40">
        <div className="flex items-center justify-between px-4 h-14">
          <h1>Your Alerts</h1>

          <div className="flex items-center gap-2">
            <button
              className="p-2 min-w-[44px] min-h-[44px] flex items-center justify-center"
              aria-label="Filter alerts"
            >
              <Filter className="w-5 h-5" />
            </button>
            <button
              className="p-2 min-w-[44px] min-h-[44px] flex items-center justify-center"
              aria-label="Settings"
            >
              <Settings className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Filter Chips */}
        <div className="flex gap-2 px-4 pb-3 overflow-x-auto">
          {['all', 'active', 'archived'].map((f) => (
            <Badge
              key={f}
              variant={filter === f ? 'default' : 'outline'}
              className="cursor-pointer capitalize whitespace-nowrap"
              onClick={() => setFilter(f as any)}
            >
              {f}
            </Badge>
          ))}
        </div>
      </div>

      {/* Alerts List */}
      <div className="p-4 space-y-3">
        {activeAlerts.length > 0 ? (
          <>
            {activeAlerts.map((alert) => (
              <Card
                key={alert.id}
                className={`p-4 border-l-4 ${getAlertBorderColor(alert.color)} cursor-pointer hover:shadow-md transition-shadow`}
                onClick={() => onAlertClick(alert)}
              >
                <div className="flex items-start gap-3">
                  <span className="text-3xl">{getAlertIcon(alert.level)}</span>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-start justify-between gap-2 mb-2">
                      <div>
                        <Badge 
                          variant="secondary" 
                          className={`mb-1 ${alert.color === 'red' ? 'bg-red-100 text-red-700' : alert.color === 'orange' ? 'bg-orange-100 text-orange-700' : 'bg-yellow-100 text-yellow-700'}`}
                        >
                          {alert.level.toUpperCase()}
                        </Badge>
                        <h3 className="text-sm">{alert.location}</h3>
                      </div>
                    </div>

                    <p className="text-sm mb-2">{alert.description}</p>
                    
                    <div className="flex items-center gap-2 mb-3">
                      <span className="text-xs text-gray-600">Expected in {alert.timeUntil}</span>
                      <span className="text-xs text-gray-400">•</span>
                      <span className="text-xs text-gray-600">{alert.confidence}% confidence</span>
                    </div>

                    <Progress value={alert.confidence} className="h-2 mb-3" />

                    {/* Map Thumbnail Placeholder */}
                    <div className="bg-gradient-to-br from-blue-100 to-green-100 h-24 rounded-lg mb-3 flex items-center justify-center">
                      <span className="text-xs text-gray-600">Map Preview</span>
                    </div>

                    <div className="flex gap-2">
                      <Button 
                        size="sm" 
                        className="flex-1"
                        onClick={(e) => {
                          e.stopPropagation();
                          onAlertClick(alert);
                        }}
                      >
                        View Details
                      </Button>
                      {alert.level === 'warning' && (
                        <Button 
                          size="sm" 
                          variant="outline"
                          className="flex-1"
                          onClick={(e) => e.stopPropagation()}
                        >
                          Navigate Away
                        </Button>
                      )}
                    </div>
                  </div>
                </div>
              </Card>
            ))}

            {/* Resolved Alerts */}
            {archivedAlerts.map((alert) => (
              <Card
                key={alert.id}
                className="p-4 bg-gray-50 border border-gray-200"
              >
                <div className="flex items-start gap-3">
                  <span className="text-2xl">✅</span>
                  <div className="flex-1">
                    <Badge variant="secondary" className="bg-green-100 text-green-700 mb-1">
                      RESOLVED
                    </Badge>
                    <h3 className="text-sm mb-1">{alert.location}</h3>
                    <p className="text-sm text-gray-600">{alert.description}</p>
                    <p className="text-xs text-gray-500 mt-2">Lifted at {alert.resolvedAt}</p>
                  </div>
                </div>
              </Card>
            ))}
          </>
        ) : (
          <div className="text-center py-16">
            <div className="text-6xl mb-4">🟢</div>
            <h2 className="text-xl mb-2">No Active Alerts</h2>
            <p className="text-gray-600 mb-1">All areas in your watch list are safe</p>
            <p className="text-sm text-gray-500">Last checked: 5:30 PM</p>
          </div>
        )}
      </div>
    </div>
  );
}
