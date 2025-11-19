import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import type { CityKey } from '../lib/map/cityConfigs';
import { MAP_CONSTANTS } from '../lib/map/config';

interface CityContextType {
    city: CityKey;
    setCity: (city: CityKey) => void;
}

const CityContext = createContext<CityContextType | undefined>(undefined);

const STORAGE_KEY = 'floodsafe_selected_city';

interface CityProviderProps {
    children: ReactNode;
}

export function CityProvider({ children }: CityProviderProps) {
    // Initialize from localStorage or use default
    const [city, setCityState] = useState<CityKey>(() => {
        if (typeof window !== 'undefined') {
            const saved = localStorage.getItem(STORAGE_KEY);
            if (saved && (saved === 'bangalore' || saved === 'delhi')) {
                return saved as CityKey;
            }
        }
        return MAP_CONSTANTS.DEFAULT_CITY;
    });

    // Persist to localStorage when city changes
    useEffect(() => {
        if (typeof window !== 'undefined') {
            localStorage.setItem(STORAGE_KEY, city);
        }
    }, [city]);

    const setCity = (newCity: CityKey) => {
        setCityState(newCity);
    };

    return (
        <CityContext.Provider value={{ city, setCity }}>
            {children}
        </CityContext.Provider>
    );
}

/**
 * Hook to access the current city and city setter
 * @returns Current city key and setter function
 * @throws Error if used outside CityProvider
 */
export function useCityContext(): CityContextType {
    const context = useContext(CityContext);
    if (context === undefined) {
        throw new Error('useCityContext must be used within a CityProvider');
    }
    return context;
}

/**
 * Hook to get just the current city (convenience hook)
 */
export function useCurrentCity(): CityKey {
    const { city } = useCityContext();
    return city;
}
