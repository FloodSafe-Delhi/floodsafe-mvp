import { useState, useEffect, useCallback, RefObject } from 'react';

interface AdaptiveScrollConfig {
    fixedBottomElements?: string[];
    baseBuffer?: number;
    minPadding?: number;
}

/**
 * Intelligent hook for calculating adaptive scroll padding
 *
 * Features:
 * - Uses visualViewport API for accurate mobile measurements
 * - Measures actual fixed element heights dynamically
 * - Adapts to zoom levels
 * - Responds to resize, orientation change, keyboard appearance
 * - Accounts for safe areas (iPhone notch/home indicator)
 */
export function useAdaptiveScrollPadding(
    scrollContainerRef: RefObject<HTMLDivElement>,
    config: AdaptiveScrollConfig = {}
) {
    const {
        fixedBottomElements = ['[data-action-buttons]', '[data-bottom-nav]'],
        baseBuffer = 40,
        minPadding = 200
    } = config;

    const [bottomPadding, setBottomPadding] = useState(minPadding);

    const calculatePadding = useCallback(() => {
        // Use visualViewport API for accurate mobile measurements
        // This accounts for zoom, keyboard, and browser chrome
        const viewport = window.visualViewport;
        const viewportHeight = viewport?.height || window.innerHeight;
        const viewportScale = viewport?.scale || 1;

        // Measure actual heights of fixed elements from bottom of viewport
        let totalFixedHeight = 0;
        fixedBottomElements.forEach(selector => {
            const el = document.querySelector(selector);
            if (el) {
                const rect = el.getBoundingClientRect();
                // Calculate how much space this element takes from the bottom
                const heightFromBottom = window.innerHeight - rect.top;
                totalFixedHeight = Math.max(totalFixedHeight, heightFromBottom);
            }
        });

        // Fallback if elements not found yet (initial render)
        if (totalFixedHeight === 0) {
            totalFixedHeight = 160; // 64px BottomNav + ~96px ActionButtons with padding
        }

        // Calculate zoom-adaptive buffer (more padding at higher zoom)
        const zoomBuffer = baseBuffer * Math.max(1, viewportScale);

        // Try to get safe area inset for iOS devices
        let safeAreaBottom = 0;
        try {
            const computedStyle = getComputedStyle(document.documentElement);
            const safeAreaValue = computedStyle.getPropertyValue('--sab');
            safeAreaBottom = parseFloat(safeAreaValue) || 0;
        } catch {
            safeAreaBottom = 0;
        }

        // Calculate viewport-proportional buffer (8% of viewport height)
        // This ensures content is comfortably visible above fixed elements
        const viewportBuffer = Math.max(20, viewportHeight * 0.08);

        // Final calculation
        const calculatedPadding = Math.ceil(
            totalFixedHeight + zoomBuffer + safeAreaBottom + viewportBuffer
        );

        // Use the larger of calculated or minimum
        setBottomPadding(Math.max(minPadding, calculatedPadding));
    }, [fixedBottomElements, baseBuffer, minPadding]);

    useEffect(() => {
        // Initial calculation with slight delay for DOM to be ready
        const initialTimeout = setTimeout(calculatePadding, 50);

        // VisualViewport API - handles zoom, keyboard, browser chrome
        const viewport = window.visualViewport;
        if (viewport) {
            viewport.addEventListener('resize', calculatePadding);
            viewport.addEventListener('scroll', calculatePadding);
        }

        // Fallback event listeners
        window.addEventListener('resize', calculatePadding);
        window.addEventListener('orientationchange', calculatePadding);

        // ResizeObserver for scroll container changes
        let resizeObserver: ResizeObserver | null = null;
        if (typeof ResizeObserver !== 'undefined') {
            resizeObserver = new ResizeObserver(calculatePadding);
            if (scrollContainerRef.current) {
                resizeObserver.observe(scrollContainerRef.current);
            }
            // Also observe body for overall layout changes
            resizeObserver.observe(document.body);
        }

        // Recalculate after page fully loads (fonts, images)
        window.addEventListener('load', calculatePadding);

        // Recalculate periodically for the first few seconds (catch late renders)
        const intervals = [100, 300, 500, 1000, 2000].map(delay =>
            setTimeout(calculatePadding, delay)
        );

        return () => {
            clearTimeout(initialTimeout);
            intervals.forEach(clearTimeout);
            viewport?.removeEventListener('resize', calculatePadding);
            viewport?.removeEventListener('scroll', calculatePadding);
            window.removeEventListener('resize', calculatePadding);
            window.removeEventListener('orientationchange', calculatePadding);
            window.removeEventListener('load', calculatePadding);
            resizeObserver?.disconnect();
        };
    }, [calculatePadding, scrollContainerRef]);

    return bottomPadding;
}
