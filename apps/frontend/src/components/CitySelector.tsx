import React, { useState, useEffect } from 'react';
import { MapPin, ChevronDown, Navigation2 } from 'lucide-react';
import { Button } from './ui/button';
import {
    DropdownMenu,
    DropdownMenuContent,
    DropdownMenuItem,
    DropdownMenuTrigger,
    DropdownMenuSeparator,
} from './ui/dropdown-menu';
import { CITY_CONFIGS, getCurrentCity, setCurrentCity, detectCityFromCoords, type CityConfig } from '../lib/map/cityConfigs';

interface CitySelectorProps {
    onCityChange: (cityCode: 'BLR' | 'DEL') => void;
    className?: string;
}

export function CitySelector({ onCityChange, className = '' }: CitySelectorProps) {
    const [selectedCity, setSelectedCityState] = useState<CityConfig>(getCurrentCity());
    const [isDetecting, setIsDetecting] = useState(false);

    const handleCityChange = (cityCode: 'BLR' | 'DEL') => {
        const newCity = CITY_CONFIGS[cityCode];
        setSelectedCityState(newCity);
        setCurrentCity(cityCode);
        onCityChange(cityCode);
    };

    const handleAutoDetect = () => {
        setIsDetecting(true);

        if (!navigator.geolocation) {
            alert('Geolocation is not supported by your browser');
            setIsDetecting(false);
            return;
        }

        navigator.geolocation.getCurrentPosition(
            (position) => {
                const { longitude, latitude } = position.coords;
                const detectedCity = detectCityFromCoords(longitude, latitude);

                if (detectedCity) {
                    handleCityChange(detectedCity);
                } else {
                    alert('Your location is not in a supported city. Please select manually.');
                }
                setIsDetecting(false);
            },
            (error) => {
                console.error('Geolocation error:', error);
                let message = 'Unable to detect location. ';

                switch (error.code) {
                    case error.PERMISSION_DENIED:
                        message += 'Location permission denied.';
                        break;
                    case error.POSITION_UNAVAILABLE:
                        message += 'Location information unavailable.';
                        break;
                    case error.TIMEOUT:
                        message += 'Location request timed out.';
                        break;
                    default:
                        message += 'Unknown error occurred.';
                }

                alert(message + ' Please select city manually.');
                setIsDetecting(false);
            },
            {
                enableHighAccuracy: true,
                timeout: 10000,
                maximumAge: 300000 // Cache for 5 minutes
            }
        );
    };

    return (
        <div className={`flex items-center gap-2 ${className}`}>
            <DropdownMenu>
                <DropdownMenuTrigger asChild>
                    <Button variant="outline" className="gap-2" disabled={isDetecting}>
                        <MapPin className="h-4 w-4" />
                        <span className="hidden sm:inline">{selectedCity.displayName}</span>
                        <span className="sm:hidden">{selectedCity.name}</span>
                        <ChevronDown className="h-4 w-4 opacity-50" />
                    </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="start">
                    {Object.values(CITY_CONFIGS).map((city) => (
                        <DropdownMenuItem
                            key={city.code}
                            onClick={() => handleCityChange(city.code)}
                            className={selectedCity.code === city.code ? 'bg-accent' : ''}
                        >
                            <div className="flex flex-col">
                                <span className="font-medium">{city.name}</span>
                                <span className="text-xs text-muted-foreground">{city.displayName}</span>
                            </div>
                        </DropdownMenuItem>
                    ))}
                    <DropdownMenuSeparator />
                    <DropdownMenuItem onClick={handleAutoDetect} disabled={isDetecting}>
                        <Navigation2 className="mr-2 h-4 w-4" />
                        <span>{isDetecting ? 'Detecting...' : 'Auto-detect location'}</span>
                    </DropdownMenuItem>
                </DropdownMenuContent>
            </DropdownMenu>
        </div>
    );
}
