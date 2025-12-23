/**
 * Mock Geolocation Provider for Testing Navigation
 *
 * This module replaces navigator.geolocation with a mock that can be
 * controlled programmatically. It allows testing GPS-dependent features
 * without physical device movement.
 *
 * IMPORTANT: Must be installed before any component calls geolocation APIs.
 */

export interface MockPosition {
    lat: number;
    lng: number;
    accuracy?: number;
    speed?: number;
    heading?: number;
}

type PositionCallback = (position: GeolocationPosition) => void;
type ErrorCallback = (error: GeolocationPositionError) => void;

interface WatchEntry {
    id: number;
    successCallback: PositionCallback;
    errorCallback?: ErrorCallback;
    options?: PositionOptions;
}

// Global state for the mock
let isInstalled = false;
let originalGeolocation: Geolocation | null = null;
let currentPosition: MockPosition | null = null;
let watchIdCounter = 1;
const watchers = new Map<number, WatchEntry>();
let positionListeners: Array<(pos: MockPosition) => void> = [];

/**
 * Convert our MockPosition to a GeolocationPosition
 */
function toGeolocationPosition(pos: MockPosition): GeolocationPosition {
    const coords: GeolocationCoordinates = {
        latitude: pos.lat,
        longitude: pos.lng,
        accuracy: pos.accuracy ?? 10,
        altitude: null,
        altitudeAccuracy: null,
        heading: pos.heading ?? null,
        speed: pos.speed ?? null,
        toJSON() {
            return {
                latitude: this.latitude,
                longitude: this.longitude,
                accuracy: this.accuracy,
                altitude: this.altitude,
                altitudeAccuracy: this.altitudeAccuracy,
                heading: this.heading,
                speed: this.speed,
            };
        },
    };
    return {
        coords,
        timestamp: Date.now(),
        toJSON() {
            return {
                coords: coords.toJSON(),
                timestamp: this.timestamp,
            };
        },
    };
}

/**
 * The mock geolocation object that replaces navigator.geolocation
 */
const mockGeolocation: Geolocation = {
    getCurrentPosition(
        successCallback: PositionCallback,
        errorCallback?: ErrorCallback,
        _options?: PositionOptions
    ): void {
        if (currentPosition) {
            // Simulate async behavior
            setTimeout(() => {
                successCallback(toGeolocationPosition(currentPosition!));
            }, 10);
        } else if (errorCallback) {
            setTimeout(() => {
                errorCallback({
                    code: 2, // POSITION_UNAVAILABLE
                    message: 'Mock geolocation: No position set',
                    PERMISSION_DENIED: 1,
                    POSITION_UNAVAILABLE: 2,
                    TIMEOUT: 3,
                });
            }, 10);
        }
    },

    watchPosition(
        successCallback: PositionCallback,
        errorCallback?: ErrorCallback,
        options?: PositionOptions
    ): number {
        const id = watchIdCounter++;

        watchers.set(id, {
            id,
            successCallback,
            errorCallback,
            options,
        });

        // If we already have a position, send it immediately
        if (currentPosition) {
            setTimeout(() => {
                successCallback(toGeolocationPosition(currentPosition!));
            }, 10);
        }

        console.log(`[MockGeo] watchPosition registered (id: ${id}), total watchers: ${watchers.size}`);
        return id;
    },

    clearWatch(watchId: number): void {
        watchers.delete(watchId);
        console.log(`[MockGeo] clearWatch (id: ${watchId}), remaining watchers: ${watchers.size}`);
    },
};

/**
 * Install the mock geolocation, replacing navigator.geolocation
 *
 * Call this at app startup when testing mode is enabled.
 */
export function installMockGeolocation(): void {
    if (isInstalled) {
        console.warn('[MockGeo] Already installed');
        return;
    }

    if (typeof navigator === 'undefined') {
        console.warn('[MockGeo] navigator not available (SSR?)');
        return;
    }

    // Store original
    originalGeolocation = navigator.geolocation;

    // Replace with mock
    Object.defineProperty(navigator, 'geolocation', {
        value: mockGeolocation,
        writable: true,
        configurable: true,
    });

    isInstalled = true;
    console.log('[MockGeo] Installed - navigator.geolocation is now mocked');
}

/**
 * Uninstall the mock and restore original geolocation
 */
export function uninstallMockGeolocation(): void {
    if (!isInstalled || !originalGeolocation) {
        console.warn('[MockGeo] Not installed, nothing to uninstall');
        return;
    }

    Object.defineProperty(navigator, 'geolocation', {
        value: originalGeolocation,
        writable: true,
        configurable: true,
    });

    // Clear state
    watchers.clear();
    currentPosition = null;
    positionListeners = [];
    isInstalled = false;

    console.log('[MockGeo] Uninstalled - navigator.geolocation restored');
}

/**
 * Set the current mock position and notify all watchers
 *
 * This is the main function to call when simulating GPS movement.
 */
export function setMockPosition(position: MockPosition): void {
    currentPosition = position;

    // Notify all watchers
    const geoPosition = toGeolocationPosition(position);
    watchers.forEach((watcher) => {
        try {
            watcher.successCallback(geoPosition);
        } catch (error) {
            console.error('[MockGeo] Error in watcher callback:', error);
        }
    });

    // Notify position listeners (for GPS test panel)
    positionListeners.forEach((listener) => {
        try {
            listener(position);
        } catch (error) {
            console.error('[MockGeo] Error in position listener:', error);
        }
    });
}

/**
 * Get the current mock position
 */
export function getMockPosition(): MockPosition | null {
    return currentPosition;
}

/**
 * Check if mock geolocation is installed
 */
export function isMockGeolocationInstalled(): boolean {
    return isInstalled;
}

/**
 * Add a listener for position updates (used by GPS test panel)
 */
export function addPositionListener(listener: (pos: MockPosition) => void): () => void {
    positionListeners.push(listener);
    return () => {
        positionListeners = positionListeners.filter((l) => l !== listener);
    };
}

/**
 * Get the number of active watchers
 */
export function getActiveWatcherCount(): number {
    return watchers.size;
}

/**
 * Simulate a geolocation error for all watchers
 */
export function simulateError(code: 1 | 2 | 3 = 2): void {
    const error: GeolocationPositionError = {
        code,
        message: code === 1 ? 'Permission denied' : code === 2 ? 'Position unavailable' : 'Timeout',
        PERMISSION_DENIED: 1,
        POSITION_UNAVAILABLE: 2,
        TIMEOUT: 3,
    };

    watchers.forEach((watcher) => {
        if (watcher.errorCallback) {
            try {
                watcher.errorCallback(error);
            } catch (err) {
                console.error('[MockGeo] Error in error callback:', err);
            }
        }
    });
}
