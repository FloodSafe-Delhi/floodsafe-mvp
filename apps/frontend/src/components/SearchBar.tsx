import { useState, useEffect, useRef, useCallback } from 'react';
import { Search, X, MapPin, Loader2 } from 'lucide-react';
import { useGeocode } from '../lib/api/hooks';
import { isWithinCityBounds, getCityConfig, type CityKey } from '../lib/map/cityConfigs';
import { GeocodingResult } from '../types';

interface SearchBarProps {
    onLocationSelect: (lat: number, lng: number, name: string) => void;
    cityKey: CityKey;
    placeholder?: string;
    className?: string;
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

export default function SearchBar({
    onLocationSelect,
    cityKey,
    placeholder = 'Search for a location...',
    className = ''
}: SearchBarProps) {
    const [query, setQuery] = useState('');
    const [isOpen, setIsOpen] = useState(false);
    const [selectedIndex, setSelectedIndex] = useState(-1);
    const inputRef = useRef<HTMLInputElement>(null);
    const dropdownRef = useRef<HTMLDivElement>(null);
    const containerRef = useRef<HTMLDivElement>(null);

    // Debounce the search query (300ms)
    const debouncedQuery = useDebounce(query, 300);

    // Use the geocode hook with debounced query
    const { data: results, isLoading, isFetching } = useGeocode(debouncedQuery, debouncedQuery.length >= 3);

    const cityConfig = getCityConfig(cityKey);

    // Filter results to only include locations within city bounds
    const filteredResults = (results || []).filter((result: GeocodingResult) => {
        const lat = parseFloat(result.lat);
        const lng = parseFloat(result.lon);
        return isWithinCityBounds(lng, lat, cityKey);
    });

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

    // Open dropdown when we have results
    useEffect(() => {
        if (debouncedQuery.length >= 3) {
            setIsOpen(true);
        }
    }, [debouncedQuery]);

    // Reset selected index when results change
    useEffect(() => {
        setSelectedIndex(-1);
    }, [filteredResults]);

    // Handle keyboard navigation
    const handleKeyDown = useCallback((e: React.KeyboardEvent<HTMLInputElement>) => {
        if (!isOpen) return;

        switch (e.key) {
            case 'ArrowDown':
                e.preventDefault();
                setSelectedIndex(prev =>
                    prev < filteredResults.length - 1 ? prev + 1 : prev
                );
                break;
            case 'ArrowUp':
                e.preventDefault();
                setSelectedIndex(prev => prev > 0 ? prev - 1 : -1);
                break;
            case 'Enter':
                e.preventDefault();
                if (selectedIndex >= 0 && filteredResults[selectedIndex]) {
                    handleSelect(filteredResults[selectedIndex]);
                }
                break;
            case 'Escape':
                e.preventDefault();
                setIsOpen(false);
                setSelectedIndex(-1);
                inputRef.current?.blur();
                break;
        }
    }, [isOpen, selectedIndex, filteredResults]);

    // Handle selection of a result
    const handleSelect = (result: GeocodingResult) => {
        const lat = parseFloat(result.lat);
        const lng = parseFloat(result.lon);

        // Format a clean display name
        const displayName = formatDisplayName(result);

        // Clear the search and close dropdown
        setQuery(displayName);
        setIsOpen(false);
        setSelectedIndex(-1);

        // Notify parent component
        onLocationSelect(lat, lng, displayName);
    };

    // Format display name to be more readable
    const formatDisplayName = (result: GeocodingResult): string => {
        const parts: string[] = [];

        // Try to get the most relevant parts of the address
        if (result.address) {
            const addr = result.address;
            if (addr.road) parts.push(addr.road);
            if (addr.neighbourhood) parts.push(addr.neighbourhood);
            if (addr.suburb) parts.push(addr.suburb);
            if (addr.city || addr.town || addr.village) {
                parts.push(addr.city || addr.town || addr.village || '');
            }
        }

        if (parts.length > 0) {
            return parts.slice(0, 3).join(', ');
        }

        // Fallback to full display name, truncated
        return result.display_name.split(',').slice(0, 3).join(', ');
    };

    // Clear search
    const handleClear = () => {
        setQuery('');
        setIsOpen(false);
        setSelectedIndex(-1);
        inputRef.current?.focus();
    };

    const showLoading = isLoading || isFetching;
    const showNoResults = !showLoading && debouncedQuery.length >= 3 && filteredResults.length === 0 && results && results.length > 0;
    const showEmptySearch = !showLoading && debouncedQuery.length >= 3 && (!results || results.length === 0);

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
                    id="location-search"
                    name="location-search"
                    type="text"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    onKeyDown={handleKeyDown}
                    onFocus={() => {
                        if (debouncedQuery.length >= 3) {
                            setIsOpen(true);
                        }
                    }}
                    placeholder={placeholder}
                    className="w-full pl-11 pr-11 py-3 text-sm bg-white border border-gray-200 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent placeholder-gray-400 transition-all"
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
            {isOpen && debouncedQuery.length >= 3 && (
                <div
                    ref={dropdownRef}
                    className="absolute z-50 w-full mt-1 bg-white border border-gray-200 rounded-lg shadow-lg overflow-hidden"
                >
                    {/* Results List */}
                    {filteredResults.length > 0 && (
                        <ul className="max-h-64 overflow-y-auto">
                            {filteredResults.map((result: GeocodingResult, index: number) => (
                                <li key={`${result.lat}-${result.lon}-${index}`}>
                                    <button
                                        type="button"
                                        onClick={() => handleSelect(result)}
                                        className={`w-full px-4 py-3 flex items-start gap-3 text-left hover:bg-gray-50 transition-colors ${
                                            selectedIndex === index ? 'bg-blue-50' : ''
                                        }`}
                                    >
                                        <MapPin className="h-5 w-5 text-blue-500 mt-0.5 flex-shrink-0" />
                                        <div className="flex-1 min-w-0">
                                            <p className="text-sm font-medium text-gray-900 truncate">
                                                {formatDisplayName(result)}
                                            </p>
                                            <p className="text-xs text-gray-500 truncate mt-0.5">
                                                {result.display_name}
                                            </p>
                                        </div>
                                    </button>
                                </li>
                            ))}
                        </ul>
                    )}

                    {/* No results in city */}
                    {showNoResults && (
                        <div className="px-4 py-3 text-sm text-gray-500">
                            <p>No results found in {cityConfig.displayName}.</p>
                            <p className="text-xs mt-1">
                                Results are filtered to the current city area.
                            </p>
                        </div>
                    )}

                    {/* No results at all */}
                    {showEmptySearch && (
                        <div className="px-4 py-3 text-sm text-gray-500">
                            No locations found for "{debouncedQuery}".
                        </div>
                    )}

                    {/* Loading state */}
                    {showLoading && (
                        <div className="px-4 py-3 flex items-center gap-2 text-sm text-gray-500">
                            <Loader2 className="h-5 w-5 animate-spin" />
                            Searching...
                        </div>
                    )}

                    {/* Footer */}
                    <div className="px-4 py-2 bg-gray-50 border-t border-gray-100">
                        <p className="text-xs text-gray-400">
                            Showing results in {cityConfig.displayName}
                        </p>
                    </div>
                </div>
            )}
        </div>
    );
}
