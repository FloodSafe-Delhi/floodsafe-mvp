import React, { useRef } from 'react';
import { useMap } from './useMap';

export const MapComponent: React.FC = () => {
    const containerRef = useRef<HTMLDivElement>(null);
    const { isLoaded } = useMap(containerRef);

    return (
        <div className="relative w-full h-screen">
            <div ref={containerRef} className="w-full h-full" />
            {!isLoaded && (
                <div className="absolute inset-0 flex items-center justify-center bg-white bg-opacity-75 z-50">
                    <span className="text-xl font-semibold">Loading Map...</span>
                </div>
            )}
        </div>
    );
};
