import { useState } from 'react';
import { Bell, RefreshCw, Loader2, AlertCircle, Cloud, Newspaper, MessageCircle, Users } from 'lucide-react';
import { AlertCard } from '../AlertCard';
import { Badge } from '../ui/badge';
import { Button } from '../ui/button';
import { useUnifiedAlerts, useRefreshExternalAlerts } from '../../lib/api/hooks';
import { useCurrentCity } from '../../contexts/CityContext';
import type { AlertSourceFilter } from '../../types';
import { toast } from 'sonner';

interface AlertsScreenProps {
    onNavigateToMap?: (lat: number, lng: number) => void;
}

/**
 * Get filter display name
 */
function getFilterLabel(filter: AlertSourceFilter): string {
    switch (filter) {
        case 'all':
            return 'All';
        case 'official':
            return 'Official';
        case 'news':
            return 'News';
        case 'social':
            return 'Social';
        case 'community':
            return 'Community';
        default:
            return filter;
    }
}

/**
 * Get filter icon
 */
function getFilterIcon(filter: AlertSourceFilter) {
    switch (filter) {
        case 'official':
            return <Cloud className="w-3 h-3" />;
        case 'news':
            return <Newspaper className="w-3 h-3" />;
        case 'social':
            return <MessageCircle className="w-3 h-3" />;
        case 'community':
            return <Users className="w-3 h-3" />;
        default:
            return null;
    }
}

export function AlertsScreen({ onNavigateToMap }: AlertsScreenProps) {
    const city = useCurrentCity();
    const [sourceFilter, setSourceFilter] = useState<AlertSourceFilter>('all');

    // Fetch alerts
    const { data, isLoading, error, refetch } = useUnifiedAlerts(city, sourceFilter);
    const refreshMutation = useRefreshExternalAlerts(city);

    // Handle manual refresh
    const handleRefresh = async () => {
        try {
            await refreshMutation.mutateAsync();
            await refetch();
            toast.success('Alerts refreshed successfully');
        } catch (err) {
            toast.error('Failed to refresh alerts. Please try again.');
        }
    };

    // Filter tabs with counts
    const filters: AlertSourceFilter[] = ['all', 'official', 'news', 'social', 'community'];

    // Get count for each filter
    const getFilterCount = (filter: AlertSourceFilter): number => {
        if (!data) return 0;
        if (filter === 'all') return data.total;

        // Map filters to source types
        const sourceMapping: Record<AlertSourceFilter, string[]> = {
            all: [],
            official: ['imd', 'cwc'],
            news: ['rss'],
            social: ['twitter', 'telegram'],
            community: ['floodsafe'],
        };

        const sources = sourceMapping[filter] || [];
        return data.alerts.filter(alert => sources.includes(alert.source)).length;
    };

    // Loading state
    if (isLoading) {
        return (
            <div className="pb-16 min-h-screen bg-gray-50 flex items-center justify-center">
                <div className="flex flex-col items-center gap-3">
                    <Loader2 className="w-8 h-8 animate-spin text-blue-600" />
                    <p className="text-gray-600">Loading alerts...</p>
                </div>
            </div>
        );
    }

    // Error state
    if (error) {
        return (
            <div className="pb-16 min-h-screen bg-gray-50 p-4">
                <div className="flex flex-col items-center justify-center py-16">
                    <AlertCircle className="w-12 h-12 text-red-500 mb-4" />
                    <h2 className="text-xl font-semibold mb-2">Failed to Load Alerts</h2>
                    <p className="text-gray-600 mb-4">Please check your connection and try again.</p>
                    <Button onClick={() => refetch()} variant="outline">
                        <RefreshCw className="w-4 h-4 mr-2" />
                        Retry
                    </Button>
                </div>
            </div>
        );
    }

    const alerts = data?.alerts || [];
    const sources = data?.sources || {};

    // Build source summary text
    const sourceSummary = Object.entries(sources)
        .filter(([_, meta]) => meta.enabled && meta.count > 0)
        .map(([_, meta]) => `${meta.name} (${meta.count})`)
        .join(' • ');

    return (
        <div className="pb-16 min-h-screen bg-gray-50">
            {/* Header */}
            <div className="bg-white shadow-sm sticky top-14 z-40">
                <div className="flex items-center justify-between px-4 h-14">
                    <div className="flex items-center gap-2">
                        <Bell className="w-5 h-5 text-blue-600" />
                        <h1 className="font-semibold">Alerts</h1>
                        <Badge variant="outline" className="ml-1">
                            {city === 'delhi' ? 'Delhi' : 'Bangalore'}
                        </Badge>
                    </div>
                    <Button
                        variant="ghost"
                        size="sm"
                        onClick={handleRefresh}
                        disabled={refreshMutation.isPending}
                    >
                        <RefreshCw className={`w-4 h-4 ${refreshMutation.isPending ? 'animate-spin' : ''}`} />
                    </Button>
                </div>

                {/* Filter Tabs */}
                <div className="flex gap-2 px-4 pb-3 overflow-x-auto scrollbar-hide">
                    {filters.map((filter) => {
                        const count = getFilterCount(filter);
                        const isActive = sourceFilter === filter;

                        return (
                            <Badge
                                key={filter}
                                variant={isActive ? 'default' : 'outline'}
                                className="cursor-pointer capitalize flex-shrink-0 px-3 py-1.5"
                                onClick={() => setSourceFilter(filter)}
                            >
                                {getFilterIcon(filter)}
                                <span className="ml-1">{getFilterLabel(filter)}</span>
                                {count > 0 && (
                                    <span className={`ml-1.5 ${isActive ? 'opacity-90' : 'opacity-60'}`}>
                                        {count}
                                    </span>
                                )}
                            </Badge>
                        );
                    })}
                </div>

                {/* Source Summary */}
                {sourceSummary && (
                    <div className="px-4 pb-3 text-xs text-gray-500 border-t pt-2">
                        <span className="font-medium">Sources: </span>
                        {sourceSummary}
                    </div>
                )}
            </div>

            {/* Alerts List */}
            <div className="p-4 space-y-3">
                {alerts.length > 0 ? (
                    alerts.map((alert) => (
                        <AlertCard
                            key={alert.id}
                            alert={alert}
                            onViewOnMap={onNavigateToMap}
                        />
                    ))
                ) : (
                    <div className="text-center py-16">
                        <div className="w-16 h-16 rounded-full bg-blue-100 flex items-center justify-center mx-auto mb-4">
                            <Bell className="w-8 h-8 text-blue-600" />
                        </div>
                        <h2 className="text-xl font-medium mb-2">No Alerts</h2>
                        <p className="text-gray-600">
                            {sourceFilter === 'all'
                                ? 'No active alerts in your area'
                                : `No ${getFilterLabel(sourceFilter).toLowerCase()} alerts available`}
                        </p>
                        <Button
                            variant="outline"
                            className="mt-4"
                            onClick={handleRefresh}
                            disabled={refreshMutation.isPending}
                        >
                            <RefreshCw className={`w-4 h-4 mr-2 ${refreshMutation.isPending ? 'animate-spin' : ''}`} />
                            Check for Updates
                        </Button>
                    </div>
                )}
            </div>
        </div>
    );
}
