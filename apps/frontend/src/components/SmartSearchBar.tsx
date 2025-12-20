import { useState, useEffect, useRef, useCallback } from 'react';
import { Search, X, Loader2, MapPin, FileText, User, TrendingUp, Sparkles } from 'lucide-react';
import { toast } from 'sonner';
import { useUnifiedSearch, useTrendingSearches } from '../lib/api/hooks';
import type {
    SearchLocationResult,
    SearchReportResult,
    SearchUserResult,
    SearchIntent
} from '../types';

interface SmartSearchBarProps {
    onLocationSelect?: (lat: number, lng: number, name: string) => void;
    onReportSelect?: (report: SearchReportResult) => void;
    onUserSelect?: (user: SearchUserResult) => void;
    placeholder?: string;
    className?: string;
    showTrending?: boolean;
}

/**
 * Custom hook for debouncing a value
 */
function useDebounce<T>(value: T, delay: number): T {
    const [debouncedValue, setDebouncedValue] = useState<T>(value);

    useEffect(() => {
        const handler = setTimeout(() => {
            setDebouncedValue(value);
        }, delay);

        return () => {
            clearTimeout(handler);
        };
    }, [value, delay]);

    return debouncedValue;
}

export default function SmartSearchBar({
    onLocationSelect,
    onReportSelect,
    onUserSelect,
    placeholder = 'Search locations, reports, or users...',
    className = '',
    showTrending = true
}: SmartSearchBarProps) {
    const [query, setQuery] = useState('');
    const [isOpen, setIsOpen] = useState(false);
    const [selectedIndex, setSelectedIndex] = useState(-1);
    const inputRef = useRef<HTMLInputElement>(null);
    const containerRef = useRef<HTMLDivElement>(null);

    // Debounce the search query (100ms for near-instant response)
    const debouncedQuery = useDebounce(query, 100);

    // Use the unified search hook
    const { data: searchResults, isLoading, isFetching } = useUnifiedSearch({
        query: debouncedQuery,
        enabled: debouncedQuery.length >= 2
    });

    // Get trending searches for empty state
    const { data: trending } = useTrendingSearches(5);

    // Deduplicate locations by formatted_name + coordinates to avoid duplicate key warnings
    const deduplicatedLocations = searchResults?.locations
        ? searchResults.locations.filter((loc, index, self) =>
            index === self.findIndex((l) =>
                l.formatted_name === loc.formatted_name &&
                Math.abs((l.lat || 0) - (loc.lat || 0)) < 0.001 &&
                Math.abs((l.lng || 0) - (loc.lng || 0)) < 0.001
            )
        )
        : [];

    // Flatten all results for keyboard navigation
    const allResults = [
        ...deduplicatedLocations,
        ...(searchResults?.reports || []),
        ...(searchResults?.users || [])
    ];

    // Handle click outside to close dropdown
    useEffect(() => {
        function handleClickOutside(event: MouseEvent) {
            if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
                setIsOpen(false);
                setSelectedIndex(-1);
            }
        }

        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, []);

    // Open dropdown only when typing (not on mount)
    useEffect(() => {
        if (debouncedQuery.length >= 2) {
            setIsOpen(true);
        }
    }, [debouncedQuery]);

    // Reset selected index when results change
    useEffect(() => {
        setSelectedIndex(-1);
    }, [searchResults]);

    // Handle keyboard navigation
    const handleKeyDown = useCallback((e: React.KeyboardEvent<HTMLInputElement>) => {
        if (!isOpen) return;

        switch (e.key) {
            case 'ArrowDown':
                e.preventDefault();
                setSelectedIndex(prev =>
                    prev < allResults.length - 1 ? prev + 1 : prev
                );
                break;
            case 'ArrowUp':
                e.preventDefault();
                setSelectedIndex(prev => prev > 0 ? prev - 1 : -1);
                break;
            case 'Enter':
                e.preventDefault();
                if (selectedIndex >= 0 && allResults[selectedIndex]) {
                    handleSelect(allResults[selectedIndex]);
                }
                break;
            case 'Escape':
                e.preventDefault();
                setIsOpen(false);
                setSelectedIndex(-1);
                inputRef.current?.blur();
                break;
        }
    }, [isOpen, selectedIndex, allResults]);

    // Handle selection of a result
    const handleSelect = (result: SearchLocationResult | SearchReportResult | SearchUserResult) => {
        console.log('[SmartSearchBar] Selected:', result);

        if (result.type === 'location') {
            const loc = result as SearchLocationResult;
            // Add null checks for lat/lng
            if (onLocationSelect && loc.lat !== undefined && loc.lng !== undefined) {
                onLocationSelect(loc.lat, loc.lng, loc.formatted_name || loc.display_name);
                setQuery(loc.formatted_name || loc.display_name);
                toast.success(`Location set: ${loc.formatted_name || loc.display_name}`, { duration: 2000 });
            } else {
                console.warn('[SmartSearchBar] Location missing lat/lng:', loc);
                toast.error('Invalid location data');
            }
        } else if (result.type === 'report' && onReportSelect) {
            onReportSelect(result as SearchReportResult);
            setQuery(result.description.substring(0, 50));
            toast.info('Report selected', { duration: 2000 });
        } else if (result.type === 'user' && onUserSelect) {
            onUserSelect(result as SearchUserResult);
            setQuery(result.username);
            toast.info(`Viewing @${result.username}`, { duration: 2000 });
        }

        setIsOpen(false);
        setSelectedIndex(-1);
    };

    // Clear search
    const handleClear = () => {
        setQuery('');
        setIsOpen(false);
        setSelectedIndex(-1);
        inputRef.current?.focus();
    };

    // Handle trending click
    const handleTrendingClick = (term: string) => {
        setQuery(term);
        inputRef.current?.focus();
    };

    const showLoading = isLoading || isFetching;
    const showResults = debouncedQuery.length >= 2 && searchResults;
    const showTrendingSection = query.length === 0 && showTrending && trending;
    const hasResults = deduplicatedLocations.length +
                      (searchResults?.reports.length || 0) +
                      (searchResults?.users.length || 0) > 0;

    // Get intent badge
    const getIntentBadge = (intent: SearchIntent) => {
        const badges = {
            location: { icon: MapPin, label: 'Location', color: 'text-blue-600 bg-blue-50' },
            report: { icon: FileText, label: 'Report', color: 'text-orange-600 bg-orange-50' },
            user: { icon: User, label: 'User', color: 'text-purple-600 bg-purple-50' },
            mixed: { icon: Sparkles, label: 'Smart', color: 'text-green-600 bg-green-50' },
            empty: { icon: Search, label: 'Search', color: 'text-gray-600 bg-gray-50' }
        };
        return badges[intent] || badges.empty;
    };

    const intentBadge = searchResults ? getIntentBadge(searchResults.intent) : null;

    return (
        <div ref={containerRef} className={`relative ${className}`}>
            {/* Search Input */}
            <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                    {showLoading ? (
                        <Loader2 className="h-5 w-5 text-gray-400 animate-spin" />
                    ) : (
                        <Search className="h-5 w-5 text-gray-400" />
                    )}
                </div>
                <input
                    ref={inputRef}
                    type="text"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    onKeyDown={handleKeyDown}
                    onFocus={() => {
                        // Only show dropdown if there's already a query
                        if (query.length >= 2) {
                            setIsOpen(true);
                        }
                    }}
                    placeholder={placeholder}
                    className="w-full pl-11 pr-11 py-3 text-sm font-normal bg-white border border-gray-200 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent placeholder:text-muted-foreground transition-all font-sans"
                    autoComplete="off"
                    spellCheck="false"
                />
                {query && (
                    <button
                        type="button"
                        onClick={handleClear}
                        className="absolute inset-y-0 right-0 pr-3 flex items-center hover:opacity-70 transition-opacity"
                    >
                        <X className="h-5 w-5 text-gray-400" />
                    </button>
                )}
            </div>

            {/* Dropdown Results */}
            {isOpen && (
                <div className="absolute z-50 w-full mt-1 bg-white border border-gray-200 rounded-lg shadow-lg overflow-hidden max-h-80 sm:max-h-96 flex flex-col font-sans">

                    {/* Dropdown Header with Close Button */}
                    <div className="px-3 py-2 bg-gray-50 border-b border-gray-200 flex items-center justify-between flex-shrink-0">
                        <span className="text-xs font-medium text-gray-600 tracking-wide">
                            {showResults ? 'Search Results' : 'Suggestions'}
                        </span>
                        <button
                            type="button"
                            onClick={() => setIsOpen(false)}
                            className="p-1 hover:bg-gray-200 rounded transition-colors"
                            aria-label="Close suggestions"
                        >
                            <X className="h-4 w-4 text-gray-500" />
                        </button>
                    </div>

                    {/* Scrollable Content */}
                    <div className="overflow-y-auto flex-1">
                        {/* Intent Badge */}
                        {intentBadge && showResults && (
                            <div className="px-4 py-2 bg-gray-50 border-b border-gray-100 flex items-center gap-2">
                                <intentBadge.icon className={`h-4 w-4 ${intentBadge.color.split(' ')[0]}`} />
                                <span className={`text-xs font-medium px-2 py-0.5 rounded ${intentBadge.color}`}>
                                    {intentBadge.label} Search
                                </span>
                            </div>
                        )}

                    {/* Locations */}
                    {showResults && deduplicatedLocations.length > 0 && (
                        <div>
                            <div className="px-4 py-2 text-xs font-semibold text-gray-500 uppercase bg-gray-50">
                                Locations
                            </div>
                            {deduplicatedLocations.map((location, index) => (
                                <button
                                    key={`loc-${index}-${location.lat?.toFixed(4) || 'na'}-${location.lng?.toFixed(4) || 'na'}`}
                                    type="button"
                                    onClick={() => handleSelect(location)}
                                    className={`w-full px-4 py-3 flex items-start gap-3 text-left hover:bg-gray-50 transition-colors ${
                                        selectedIndex === index ? 'bg-blue-50' : ''
                                    }`}
                                >
                                    <MapPin className="h-5 w-5 text-blue-500 mt-0.5 flex-shrink-0" />
                                    <div className="flex-1 min-w-0">
                                        <p className="text-sm font-medium text-gray-900 truncate">
                                            {location.formatted_name}
                                        </p>
                                        <p className="text-xs text-gray-500 truncate mt-0.5">
                                            {location.display_name}
                                        </p>
                                    </div>
                                </button>
                            ))}
                        </div>
                    )}

                    {/* Reports */}
                    {showResults && searchResults.reports.length > 0 && (
                        <div>
                            <div className="px-4 py-2 text-xs font-semibold text-gray-500 uppercase bg-gray-50 border-t border-gray-100">
                                Flood Reports
                            </div>
                            {searchResults.reports.map((report, index) => {
                                const resultIndex = deduplicatedLocations.length + index;
                                return (
                                    <button
                                        key={`report-${report.id}`}
                                        type="button"
                                        onClick={() => handleSelect(report)}
                                        className={`w-full px-4 py-3 flex items-start gap-3 text-left hover:bg-gray-50 transition-colors ${
                                            selectedIndex === resultIndex ? 'bg-blue-50' : ''
                                        }`}
                                    >
                                        <FileText className="h-5 w-5 text-orange-500 mt-0.5 flex-shrink-0" />
                                        <div className="flex-1 min-w-0">
                                            <p className="text-sm text-gray-900">
                                                {report.highlight}
                                            </p>
                                            <div className="flex items-center gap-2 mt-1">
                                                {report.verified && (
                                                    <span className="text-xs px-1.5 py-0.5 bg-green-100 text-green-700 rounded">
                                                        Verified
                                                    </span>
                                                )}
                                                {report.water_depth && (
                                                    <span className="text-xs text-gray-500">
                                                        {report.water_depth} deep
                                                    </span>
                                                )}
                                            </div>
                                        </div>
                                    </button>
                                );
                            })}
                        </div>
                    )}

                    {/* Users */}
                    {showResults && searchResults.users.length > 0 && (
                        <div>
                            <div className="px-4 py-2 text-xs font-semibold text-gray-500 uppercase bg-gray-50 border-t border-gray-100">
                                Users
                            </div>
                            {searchResults.users.map((user, index) => {
                                const resultIndex = deduplicatedLocations.length + searchResults.reports.length + index;
                                return (
                                    <button
                                        key={`user-${user.id}`}
                                        type="button"
                                        onClick={() => handleSelect(user)}
                                        className={`w-full px-4 py-3 flex items-start gap-3 text-left hover:bg-gray-50 transition-colors ${
                                            selectedIndex === resultIndex ? 'bg-blue-50' : ''
                                        }`}
                                    >
                                        <User className="h-5 w-5 text-purple-500 mt-0.5 flex-shrink-0" />
                                        <div className="flex-1 min-w-0">
                                            <p className="text-sm font-medium text-gray-900">
                                                @{user.username}
                                            </p>
                                            <div className="flex items-center gap-2 mt-1">
                                                <span className="text-xs text-gray-500">
                                                    Level {user.level} • {user.points} pts
                                                </span>
                                                {user.reports_count > 0 && (
                                                    <span className="text-xs text-gray-500">
                                                        • {user.reports_count} reports
                                                    </span>
                                                )}
                                            </div>
                                        </div>
                                    </button>
                                );
                            })}
                        </div>
                    )}

                    {/* No Results */}
                    {showResults && !hasResults && (
                        <div className="px-4 py-8 text-center">
                            <Search className="h-12 w-12 text-gray-300 mx-auto mb-3" />
                            <p className="text-sm text-gray-500">
                                No results for "{debouncedQuery}"
                            </p>
                            <p className="text-xs text-gray-400 mt-1">
                                Try different keywords or check spelling
                            </p>
                        </div>
                    )}

                    {/* Trending Searches (Empty State) */}
                    {showTrendingSection && (
                        <div>
                            <div className="px-4 py-2 text-xs font-semibold text-gray-500 uppercase bg-gray-50 flex items-center gap-2">
                                <TrendingUp className="h-4 w-4" />
                                Trending Searches
                            </div>
                            <div className="px-4 py-3 flex flex-wrap gap-2">
                                {trending.trending.map((term, index) => (
                                    <button
                                        key={index}
                                        type="button"
                                        onClick={() => handleTrendingClick(term)}
                                        className="px-3 py-1.5 text-sm bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-full transition-colors"
                                    >
                                        {term}
                                    </button>
                                ))}
                            </div>
                        </div>
                    )}

                    {/* Loading State */}
                    {showLoading && (
                        <div className="px-4 py-3 flex items-center gap-2 text-sm text-gray-500">
                            <Loader2 className="h-5 w-5 animate-spin" />
                            Searching...
                        </div>
                    )}

                    {/* Search Tips */}
                    {showResults && searchResults.suggestions && searchResults.suggestions.length > 0 && (
                        <div className="px-4 py-3 bg-blue-50 border-t border-blue-100">
                            {searchResults.suggestions.map((suggestion, index) => (
                                <div key={index} className="text-xs text-blue-700">
                                    {suggestion.text}
                                </div>
                            ))}
                        </div>
                    )}
                    </div>
                </div>
            )}
        </div>
    );
}
