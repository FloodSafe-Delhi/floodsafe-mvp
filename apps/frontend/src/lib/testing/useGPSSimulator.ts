/**
 * GPS Simulator Hook for Testing Navigation Without Physical Movement
 *
 * This utility simulates GPS movement along a predefined route,
 * allowing testing of turn-by-turn navigation, deviation detection,
 * and hotspot proximity alerts without being in Delhi or moving physically.
 */

import { useState, useCallback, useRef, useEffect } from 'react';
import { haversineDistance } from '../geo/distance';

export interface GPSPosition {
    lat: number;
    lng: number;
    accuracy?: number;
    speed?: number;
    heading?: number;
}

export interface GPSSimulatorOptions {
    /** Route waypoints in [lng, lat] format (GeoJSON standard) */
    route: [number, number][];
    /** Simulated speed in km/h (default: 30) */
    speedKmh?: number;
    /** Update interval in milliseconds (default: 1000) */
    intervalMs?: number;
    /** Callback when position updates */
    onPositionUpdate?: (position: GPSPosition) => void;
    /** Callback when simulation completes */
    onComplete?: () => void;
    /** Callback when deviation is triggered */
    onDeviation?: (position: GPSPosition) => void;
}

export interface GPSSimulator {
    /** Start the simulation */
    start: () => void;
    /** Stop the simulation */
    stop: () => void;
    /** Pause the simulation */
    pause: () => void;
    /** Resume the simulation */
    resume: () => void;
    /** Jump to a specific waypoint index */
    jumpToIndex: (index: number) => void;
    /** Trigger an off-route deviation */
    triggerDeviation: (offsetMeters?: number) => void;
    /** Reset deviation and return to route */
    resetDeviation: () => void;
    /** Set simulated speed */
    setSpeed: (speedKmh: number) => void;
    /** Current simulated position */
    currentPosition: GPSPosition | null;
    /** Current waypoint index */
    currentIndex: number;
    /** Progress percentage (0-100) */
    progress: number;
    /** Whether simulation is running */
    isRunning: boolean;
    /** Whether simulation is paused */
    isPaused: boolean;
    /** Whether currently in deviated state */
    isDeviated: boolean;
    /** Speed in m/s */
    speedMs: number;
}

/**
 * Interpolate between two points based on fraction
 */
function interpolatePosition(
    from: [number, number],
    to: [number, number],
    fraction: number
): GPSPosition {
    const lng = from[0] + (to[0] - from[0]) * fraction;
    const lat = from[1] + (to[1] - from[1]) * fraction;

    // Calculate heading (bearing) from from to to
    const dLng = to[0] - from[0];
    const y = Math.sin(dLng * Math.PI / 180) * Math.cos(to[1] * Math.PI / 180);
    const x = Math.cos(from[1] * Math.PI / 180) * Math.sin(to[1] * Math.PI / 180) -
              Math.sin(from[1] * Math.PI / 180) * Math.cos(to[1] * Math.PI / 180) * Math.cos(dLng * Math.PI / 180);
    const heading = (Math.atan2(y, x) * 180 / Math.PI + 360) % 360;

    return { lat, lng, heading, accuracy: 10 };
}

/**
 * Apply offset perpendicular to the route direction
 */
function applyDeviation(
    position: GPSPosition,
    offsetMeters: number
): GPSPosition {
    // Convert offset to degrees (approximate)
    const offsetDegrees = offsetMeters / 111320; // 1 degree ≈ 111.32 km at equator

    // Apply offset perpendicular to heading
    const headingRad = ((position.heading || 0) + 90) * Math.PI / 180;
    const offsetLat = offsetDegrees * Math.cos(headingRad);
    const offsetLng = offsetDegrees * Math.sin(headingRad) / Math.cos(position.lat * Math.PI / 180);

    return {
        ...position,
        lat: position.lat + offsetLat,
        lng: position.lng + offsetLng,
    };
}

export function useGPSSimulator(options: GPSSimulatorOptions): GPSSimulator {
    const {
        route,
        speedKmh = 30,
        intervalMs = 1000,
        onPositionUpdate,
        onComplete,
        onDeviation,
    } = options;

    const [currentPosition, setCurrentPosition] = useState<GPSPosition | null>(null);
    const [currentIndex, setCurrentIndex] = useState(0);
    const [isRunning, setIsRunning] = useState(false);
    const [isPaused, setIsPaused] = useState(false);
    const [isDeviated, setIsDeviated] = useState(false);
    const [speedMs, setSpeedMs] = useState((speedKmh * 1000) / 3600);

    const intervalRef = useRef<NodeJS.Timeout | null>(null);
    const progressRef = useRef(0); // Progress within current segment (0-1)
    const deviationOffsetRef = useRef(0);

    // Calculate total route distance
    const totalDistance = useRef(0);
    useEffect(() => {
        let total = 0;
        for (let i = 0; i < route.length - 1; i++) {
            total += haversineDistance(
                route[i][1], route[i][0],
                route[i + 1][1], route[i + 1][0]
            );
        }
        totalDistance.current = total;
    }, [route]);

    // Calculate overall progress percentage
    const calculateProgress = useCallback(() => {
        if (route.length < 2) return 100;

        let distanceCovered = 0;
        for (let i = 0; i < currentIndex && i < route.length - 1; i++) {
            distanceCovered += haversineDistance(
                route[i][1], route[i][0],
                route[i + 1][1], route[i + 1][0]
            );
        }

        // Add partial segment
        if (currentIndex < route.length - 1) {
            const segmentDist = haversineDistance(
                route[currentIndex][1], route[currentIndex][0],
                route[currentIndex + 1][1], route[currentIndex + 1][0]
            );
            distanceCovered += segmentDist * progressRef.current;
        }

        return totalDistance.current > 0
            ? Math.min(100, (distanceCovered / totalDistance.current) * 100)
            : 100;
    }, [route, currentIndex]);

    const updatePosition = useCallback(() => {
        if (route.length < 2 || currentIndex >= route.length - 1) {
            setIsRunning(false);
            onComplete?.();
            return;
        }

        const from = route[currentIndex];
        const to = route[currentIndex + 1];

        // Calculate segment distance
        const segmentDistance = haversineDistance(from[1], from[0], to[1], to[0]);

        // Calculate how much progress per interval
        const distancePerInterval = speedMs * (intervalMs / 1000);
        const progressPerInterval = segmentDistance > 0 ? distancePerInterval / segmentDistance : 1;

        // Update progress
        progressRef.current += progressPerInterval;

        // Check if we've reached the next waypoint
        if (progressRef.current >= 1) {
            progressRef.current = 0;
            setCurrentIndex(prev => prev + 1);

            // Check if route complete
            if (currentIndex + 1 >= route.length - 1) {
                setIsRunning(false);
                onComplete?.();
                return;
            }
        }

        // Interpolate position
        let position = interpolatePosition(from, to, progressRef.current);
        position.speed = speedMs;

        // Apply deviation if active
        if (isDeviated && deviationOffsetRef.current > 0) {
            position = applyDeviation(position, deviationOffsetRef.current);
            onDeviation?.(position);
        }

        setCurrentPosition(position);
        onPositionUpdate?.(position);
    }, [route, currentIndex, speedMs, intervalMs, isDeviated, onPositionUpdate, onComplete, onDeviation]);

    const start = useCallback(() => {
        if (route.length < 2) return;

        setIsRunning(true);
        setIsPaused(false);
        setCurrentIndex(0);
        progressRef.current = 0;

        // Set initial position
        const initialPosition: GPSPosition = {
            lat: route[0][1],
            lng: route[0][0],
            accuracy: 10,
            speed: speedMs,
        };
        setCurrentPosition(initialPosition);
        onPositionUpdate?.(initialPosition);

        // Start interval
        intervalRef.current = setInterval(updatePosition, intervalMs);
    }, [route, speedMs, intervalMs, updatePosition, onPositionUpdate]);

    const stop = useCallback(() => {
        setIsRunning(false);
        setIsPaused(false);
        setCurrentIndex(0);
        progressRef.current = 0;
        setIsDeviated(false);
        deviationOffsetRef.current = 0;

        if (intervalRef.current) {
            clearInterval(intervalRef.current);
            intervalRef.current = null;
        }
    }, []);

    const pause = useCallback(() => {
        setIsPaused(true);
        if (intervalRef.current) {
            clearInterval(intervalRef.current);
            intervalRef.current = null;
        }
    }, []);

    const resume = useCallback(() => {
        if (!isRunning) return;
        setIsPaused(false);
        intervalRef.current = setInterval(updatePosition, intervalMs);
    }, [isRunning, intervalMs, updatePosition]);

    const jumpToIndex = useCallback((index: number) => {
        if (index < 0 || index >= route.length) return;
        setCurrentIndex(index);
        progressRef.current = 0;

        const position: GPSPosition = {
            lat: route[index][1],
            lng: route[index][0],
            accuracy: 10,
            speed: speedMs,
        };
        setCurrentPosition(position);
        onPositionUpdate?.(position);
    }, [route, speedMs, onPositionUpdate]);

    const triggerDeviation = useCallback((offsetMeters: number = 100) => {
        setIsDeviated(true);
        deviationOffsetRef.current = offsetMeters;

        // Immediately apply deviation to current position
        if (currentPosition) {
            const deviatedPosition = applyDeviation(currentPosition, offsetMeters);
            setCurrentPosition(deviatedPosition);
            onPositionUpdate?.(deviatedPosition);
            onDeviation?.(deviatedPosition);
        }
    }, [currentPosition, onPositionUpdate, onDeviation]);

    const resetDeviation = useCallback(() => {
        setIsDeviated(false);
        deviationOffsetRef.current = 0;
    }, []);

    const setSpeed = useCallback((newSpeedKmh: number) => {
        setSpeedMs((newSpeedKmh * 1000) / 3600);
    }, []);

    // Cleanup on unmount
    useEffect(() => {
        return () => {
            if (intervalRef.current) {
                clearInterval(intervalRef.current);
            }
        };
    }, []);

    return {
        start,
        stop,
        pause,
        resume,
        jumpToIndex,
        triggerDeviation,
        resetDeviation,
        setSpeed,
        currentPosition,
        currentIndex,
        progress: calculateProgress(),
        isRunning,
        isPaused,
        isDeviated,
        speedMs,
    };
}
