/**
 * GPS Testing Utilities
 *
 * This module provides tools for testing GPS-dependent navigation
 * features without physical movement.
 *
 * Usage:
 * 1. Set VITE_ENABLE_GPS_TESTING=true in .env
 * 2. Open Flood Atlas - GPS Test Panel appears
 * 3. Select a route and start simulation
 * 4. Use Navigation Panel to plan/start navigation
 * 5. Navigation will use mock GPS positions
 */

export { useGPSSimulator } from './useGPSSimulator';
export type { GPSPosition, GPSSimulatorOptions, GPSSimulator } from './useGPSSimulator';

export {
    installMockGeolocation,
    uninstallMockGeolocation,
    setMockPosition,
    getMockPosition,
    isMockGeolocationInstalled,
    addPositionListener,
    getActiveWatcherCount,
    simulateError,
} from './mockGeolocation';
export type { MockPosition } from './mockGeolocation';

export {
    DELHI_TEST_ROUTES,
    DELHI_LANDMARKS,
    getAllTestRoutes,
    getTestRouteById,
    getDeviationPosition,
} from './testRoutes';
export type { TestRoute } from './testRoutes';
