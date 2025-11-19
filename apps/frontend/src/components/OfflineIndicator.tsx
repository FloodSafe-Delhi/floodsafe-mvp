import { WifiOff, RefreshCw } from 'lucide-react';
import { Button } from './ui/button';

interface OfflineIndicatorProps {
    isOffline: boolean;
    lastUpdate: string;
    onRetry: () => void;
}

export function OfflineIndicator({ isOffline, lastUpdate, onRetry }: OfflineIndicatorProps) {
    if (!isOffline) return null;

    return (
        <div className="fixed bottom-20 left-4 right-4 bg-gray-900 text-white p-3 rounded-lg shadow-lg z-50 flex items-center justify-between animate-in slide-in-from-bottom-4">
            <div className="flex items-center gap-3">
                <WifiOff className="w-5 h-5 text-red-400" />
                <div>
                    <p className="text-sm font-medium">You are offline</p>
                    <p className="text-xs text-gray-400">Last updated: {lastUpdate}</p>
                </div>
            </div>
            <Button variant="secondary" size="sm" onClick={onRetry} className="h-8">
                <RefreshCw className="w-3 h-3 mr-2" />
                Retry
            </Button>
        </div>
    );
}
