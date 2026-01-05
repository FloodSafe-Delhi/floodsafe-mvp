import { useEffect, useState } from 'react';
import { useRegisterSW } from 'virtual:pwa-register/react';
import { Download, X, RefreshCw } from 'lucide-react';
import { Button } from './ui/button';

/**
 * PWAUpdateBanner - Shows a banner when a new version of the app is available
 *
 * Uses vite-plugin-pwa's useRegisterSW hook to:
 * 1. Detect when a new service worker is available
 * 2. Prompt the user to update
 * 3. Reload the page with the new version
 */
export function PWAUpdateBanner() {
    const [showBanner, setShowBanner] = useState(false);
    const [isUpdating, setIsUpdating] = useState(false);

    const {
        needRefresh: [needRefresh, setNeedRefresh],
        offlineReady: [offlineReady, setOfflineReady],
        updateServiceWorker,
    } = useRegisterSW({
        onRegisteredSW(swUrl, registration) {
            console.log('[PWA] Service worker registered:', swUrl);

            // Check for updates every hour
            if (registration) {
                setInterval(() => {
                    registration.update();
                }, 60 * 60 * 1000);
            }
        },
        onRegisterError(error) {
            console.error('[PWA] Service worker registration failed:', error);
        },
    });

    // Show banner when update is needed or offline ready
    useEffect(() => {
        if (needRefresh) {
            setShowBanner(true);
        }
    }, [needRefresh]);

    // Auto-dismiss offline ready message after 5 seconds
    useEffect(() => {
        if (offlineReady) {
            const timer = setTimeout(() => {
                setOfflineReady(false);
            }, 5000);
            return () => clearTimeout(timer);
        }
    }, [offlineReady, setOfflineReady]);

    const handleUpdate = async () => {
        setIsUpdating(true);
        try {
            await updateServiceWorker(true); // true = reload page
        } catch (error) {
            console.error('[PWA] Update failed:', error);
            setIsUpdating(false);
        }
    };

    const handleDismiss = () => {
        setShowBanner(false);
        setNeedRefresh(false);
    };

    // Show offline ready toast
    if (offlineReady) {
        return (
            <div className="fixed top-4 left-4 right-4 md:left-auto md:right-4 md:w-80 bg-green-600 text-white p-3 rounded-lg shadow-lg z-50 flex items-center justify-between animate-in slide-in-from-top-4">
                <div className="flex items-center gap-3">
                    <Download className="w-5 h-5" />
                    <div>
                        <p className="text-sm font-medium">Ready for offline use</p>
                        <p className="text-xs opacity-80">FloodSafe works without internet</p>
                    </div>
                </div>
                <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setOfflineReady(false)}
                    className="h-8 text-white hover:bg-green-700"
                >
                    <X className="w-4 h-4" />
                </Button>
            </div>
        );
    }

    // Show update available banner
    if (!showBanner || !needRefresh) return null;

    return (
        <div className="fixed top-4 left-4 right-4 md:left-auto md:right-4 md:w-96 bg-blue-600 text-white p-4 rounded-lg shadow-lg z-50 animate-in slide-in-from-top-4">
            <div className="flex items-start justify-between">
                <div className="flex items-start gap-3">
                    <div className="mt-0.5 p-2 bg-blue-700 rounded-full">
                        <RefreshCw className="w-4 h-4" />
                    </div>
                    <div>
                        <p className="font-medium">New version available</p>
                        <p className="text-sm opacity-80 mt-1">
                            A new version of FloodSafe is ready. Update now for the latest features and improvements.
                        </p>
                    </div>
                </div>
                <Button
                    variant="ghost"
                    size="sm"
                    onClick={handleDismiss}
                    className="h-8 text-white hover:bg-blue-700 -mt-1 -mr-1"
                >
                    <X className="w-4 h-4" />
                </Button>
            </div>
            <div className="flex gap-2 mt-3 ml-11">
                <Button
                    variant="secondary"
                    size="sm"
                    onClick={handleUpdate}
                    disabled={isUpdating}
                    className="bg-white text-blue-600 hover:bg-blue-50"
                >
                    {isUpdating ? (
                        <>
                            <RefreshCw className="w-3 h-3 mr-2 animate-spin" />
                            Updating...
                        </>
                    ) : (
                        <>
                            <Download className="w-3 h-3 mr-2" />
                            Update now
                        </>
                    )}
                </Button>
                <Button
                    variant="ghost"
                    size="sm"
                    onClick={handleDismiss}
                    className="text-white hover:bg-blue-700"
                >
                    Later
                </Button>
            </div>
        </div>
    );
}
