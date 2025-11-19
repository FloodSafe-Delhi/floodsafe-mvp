import { WifiOff, RefreshCw } from 'lucide-react';
import { Button } from './ui/button';
import { Card } from './ui/card';

interface OfflineIndicatorProps {
  isOffline: boolean;
  lastUpdate?: string;
  onRetry?: () => void;
}

export function OfflineIndicator({ isOffline, lastUpdate = '3:45 PM', onRetry }: OfflineIndicatorProps) {
  if (!isOffline) return null;

  return (
    <>
      {/* Banner */}
      <div className="fixed top-14 left-0 right-0 bg-orange-500 text-white px-4 py-2 z-50 text-center">
        <p className="text-sm">
          📶 Offline - Last updated {lastUpdate}
        </p>
      </div>

      {/* Full Overlay (optional - can be toggled) */}
      <div className="fixed inset-0 bg-black/50 z-[60] flex items-center justify-center p-4">
        <Card className="max-w-md w-full p-6 text-center">
          <WifiOff className="w-16 h-16 mx-auto mb-4 text-gray-400" />
          
          <h2 className="text-xl mb-2">You're Offline</h2>
          <p className="text-gray-600 text-sm mb-4">
            Showing cached data from {lastUpdate}
          </p>

          <div className="text-left mb-6 space-y-2 text-sm">
            <p className="text-gray-700">Features available offline:</p>
            <ul className="space-y-1 ml-4">
              <li className="text-green-600">✓ View last 72 hours of alerts</li>
              <li className="text-green-600">✓ Browse cached map (500m radius)</li>
              <li className="text-green-600">✓ Submit reports (syncs when online)</li>
              <li className="text-green-600">✓ View safe routes (last known)</li>
              <li className="text-red-600">✗ Real-time updates</li>
            </ul>
          </div>

          <p className="text-sm text-gray-600 mb-4">Checking for connection...</p>

          <Button onClick={onRetry} className="w-full">
            <RefreshCw className="w-4 h-4 mr-2" />
            Retry Connection
          </Button>
        </Card>
      </div>
    </>
  );
}
