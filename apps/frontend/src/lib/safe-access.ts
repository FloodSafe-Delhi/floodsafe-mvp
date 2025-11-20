/**
 * Safe nested property access helpers
 * Provides type-safe ways to access nested properties without runtime errors
 */

/**
 * Safely get a nested property value with a default fallback
 * @param obj The object to access
 * @param path Array of property names representing the path
 * @param defaultValue Default value if property doesn't exist
 * @returns The property value or default
 *
 * @example
 * const user = { profile: { name: "John" } };
 * getNestedValue(user, ['profile', 'name'], 'Unknown'); // "John"
 * getNestedValue(user, ['profile', 'age'], 0); // 0
 */
export function getNestedValue<T>(
    obj: unknown,
    path: string[],
    defaultValue: T
): T {
    if (!obj || typeof obj !== 'object') {
        return defaultValue;
    }

    let current: any = obj;

    for (const key of path) {
        if (current === null || current === undefined || typeof current !== 'object') {
            return defaultValue;
        }

        current = current[key];
    }

    return current !== undefined && current !== null ? current : defaultValue;
}

/**
 * Check if a nested property exists and is not null/undefined
 * @param obj The object to check
 * @param path Array of property names representing the path
 * @returns true if the property exists and is not null/undefined
 *
 * @example
 * const user = { profile: { name: "John" } };
 * hasNestedValue(user, ['profile', 'name']); // true
 * hasNestedValue(user, ['profile', 'age']); // false
 */
export function hasNestedValue(obj: unknown, path: string[]): boolean {
    if (!obj || typeof obj !== 'object') {
        return false;
    }

    let current: any = obj;

    for (const key of path) {
        if (current === null || current === undefined || typeof current !== 'object') {
            return false;
        }

        current = current[key];
    }

    return current !== null && current !== undefined;
}

/**
 * Safely access nested array property
 * @param obj The object containing the array
 * @param path Array of property names leading to the array
 * @returns The array or empty array if doesn't exist or isn't an array
 *
 * @example
 * const data = { results: { items: [1, 2, 3] } };
 * getNestedArray(data, ['results', 'items']); // [1, 2, 3]
 * getNestedArray(data, ['results', 'missing']); // []
 */
export function getNestedArray<T>(obj: unknown, path: string[]): T[] {
    const value = getNestedValue(obj, path, []);
    return Array.isArray(value) ? value : [];
}

/**
 * Safely access nested object property
 * @param obj The object containing the nested object
 * @param path Array of property names leading to the object
 * @returns The object or null if doesn't exist or isn't an object
 *
 * @example
 * const data = { user: { profile: { age: 25 } } };
 * getNestedObject(data, ['user', 'profile']); // { age: 25 }
 * getNestedObject(data, ['user', 'settings']); // null
 */
export function getNestedObject(obj: unknown, path: string[]): Record<string, unknown> | null {
    const value = getNestedValue(obj, path, null);
    return value !== null && typeof value === 'object' && !Array.isArray(value)
        ? value as Record<string, unknown>
        : null;
}

/**
 * Safely extract multiple nested values at once
 * @param obj The object to extract from
 * @param paths Object mapping keys to property paths
 * @returns Object with the same keys containing extracted values or defaults
 *
 * @example
 * const data = { user: { name: "John", age: 25 } };
 * extractValues(data, {
 *   name: { path: ['user', 'name'], default: 'Unknown' },
 *   age: { path: ['user', 'age'], default: 0 },
 *   city: { path: ['user', 'city'], default: 'N/A' }
 * });
 * // Returns: { name: "John", age: 25, city: "N/A" }
 */
export function extractValues<T extends Record<string, any>>(
    obj: unknown,
    paths: { [K in keyof T]: { path: string[]; default: T[K] } }
): T {
    const result = {} as T;

    for (const key in paths) {
        const { path, default: defaultValue } = paths[key];
        result[key] = getNestedValue(obj, path, defaultValue);
    }

    return result;
}

/**
 * Type guard to check if value is a valid coordinate pair
 */
export function isCoordinatePair(value: unknown): value is [number, number] {
    return (
        Array.isArray(value) &&
        value.length === 2 &&
        typeof value[0] === 'number' &&
        typeof value[1] === 'number' &&
        !isNaN(value[0]) &&
        !isNaN(value[1])
    );
}

/**
 * Type guard to check if object has required location fields
 */
export interface LocationData {
    latitude: number;
    longitude: number;
    radius_meters?: number;
}

export function hasLocationData(obj: unknown): obj is LocationData {
    if (!obj || typeof obj !== 'object') return false;

    const data = obj as any;
    return (
        typeof data.latitude === 'number' &&
        !isNaN(data.latitude) &&
        typeof data.longitude === 'number' &&
        !isNaN(data.longitude)
    );
}
