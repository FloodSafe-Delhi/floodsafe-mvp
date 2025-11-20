# Frontend Type Safety Issues Report

## Summary
Found **12 critical and moderate type safety issues** across the frontend codebase that could cause runtime errors in production.

---

## Critical Issues

### 1. Unsafe Type Assertion with `as any`
**File**: `/home/user/floodsafe-mvp/apps/frontend/src/components/MapComponent.tsx`
**Lines**: 54, 184
**Issue**: Unsafe type assertions bypassing type checking
```typescript
// Line 54
setCity(newCity as any);

// Line 184
const coordinates = (feature.geometry as any).coordinates.slice();
```
**Potential Runtime Error**: 
- Line 54: `newCity` may not be a valid CityKey, breaking type safety
- Line 184: Accessing `.coordinates` on geometry without type guard could fail if geometry structure differs

**Suggested Fix**:
```typescript
// Line 54
setCity(newCity as CityKey); // Use proper type instead of 'any'

// Line 184
if (feature.geometry && 'coordinates' in feature.geometry) {
  const coordinates = (feature.geometry as GeoJSON.Point).coordinates.slice();
}
```

---

### 2. Array Filter/Map Without Null Check
**File**: `/home/user/floodsafe-mvp/apps/frontend/src/components/screens/HomeScreen.tsx`
**Line**: 73
**Issue**: Calling `.filter().map()` on potentially undefined array
```typescript
const activeAlerts: FloodAlert[] = sensors?.filter(s => s.status !== 'active').map(s => ({...})) || [];
```
**Problem**: 
- The optional chaining `?.filter()` returns `undefined` if `sensors` is undefined
- But then `.map()` is called on the result without checking
- This could cause: "Cannot read property 'map' of undefined"

**Suggested Fix**:
```typescript
const activeAlerts: FloodAlert[] = (sensors?.filter(s => s.status !== 'active') ?? []).map(s => ({...})) || [];
// OR
const activeAlerts: FloodAlert[] = sensors 
  ? sensors.filter(s => s.status !== 'active').map(s => ({...}))
  : [];
```

---

### 3. Direct Array Access Without Bounds Check
**File**: `/home/user/floodsafe-mvp/apps/frontend/src/components/screens/HomeScreen.tsx`
**Line**: 88
**Issue**: Accessing array element without checking if array exists or has length
```typescript
const currentUser = users?.[0]; // Mock current user - in real app would come from auth
```
**Potential Problem**: 
- `users` could be undefined OR empty array
- Later at lines 90-92, accessing properties of `currentUser` without checking if it exists:
```typescript
const userImpact = {
    reports: currentUser?.reports_count || 0,
    helped: (currentUser?.reports_count || 0) * 15, // currentUser might be undefined
};
```

**Suggested Fix**:
```typescript
const currentUser = users && users.length > 0 ? users[0] : null;
// Then use proper null checks:
const userImpact = {
    reports: currentUser?.reports_count || 0,
    helped: (currentUser?.reports_count || 0) * 15,
};
```

---

### 4. Missing Type Guard for Speech Recognition Event Results
**File**: `/home/user/floodsafe-mvp/apps/frontend/src/components/screens/ReportScreen.tsx`
**Lines**: 112-118
**Issue**: Accessing `event.results[i][0]` without type checking
```typescript
for (let i = event.resultIndex; i < event.results.length; i++) {
    if (event.results[i].isFinal) {
        transcript += event.results[i][0].transcript + ' ';
    } else if (!isIOSDevice()) {
        transcript += event.results[i][0].transcript + ' ';
    }
}
```
**Potential Runtime Error**: 
- `event.results[i]` might be an empty SpeechRecognitionAlternative array
- Accessing `[0]` without checking could throw "Cannot read property 'transcript' of undefined"
- `event.resultIndex` might not exist on all platforms

**Suggested Fix**:
```typescript
for (let i = event.resultIndex || 0; i < event.results?.length || 0; i++) {
    if (event.results?.[i]?.length > 0 && event.results[i].isFinal) {
        const alternative = event.results[i][0];
        if (alternative?.transcript) {
            transcript += alternative.transcript + ' ';
        }
    }
}
```

---

### 5. Unsafe Window Property Access
**File**: `/home/user/floodsafe-mvp/apps/frontend/src/components/screens/ReportScreen.tsx`
**Line**: 88
**Issue**: Accessing global object properties without proper type checking
```typescript
const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
```
**Problem**:
- Using `as any` defeats TypeScript protection
- Could access non-existent properties
- No type information for SpeechRecognition class

**Suggested Fix**:
```typescript
// Define a proper type first
interface WindowWithSpeechRecognition extends Window {
    SpeechRecognition?: typeof SpeechRecognition;
    webkitSpeechRecognition?: typeof SpeechRecognition;
}

const SpeechRecognition = (window as WindowWithSpeechRecognition).SpeechRecognition 
    || (window as WindowWithSpeechRecognition).webkitSpeechRecognition;
```

---

### 6. Missing Optional Chaining in Map Style Access
**File**: `/home/user/floodsafe-mvp/apps/frontend/src/lib/map/useMap.ts`
**Lines**: 141-157
**Issue**: Accessing potentially undefined properties without optional chaining
```typescript
const style = map.getStyle();
if (!style || !style.sources) {
    console.log('ℹ️ Map style not ready yet');
    return;
}

const sources = style.sources;
const sourceKey = Object.keys(sources).find(key =>
    key.includes('basemap') || key.includes('openmaptiles')
);

if (sourceKey && map.getSource(sourceKey)) {
    const features = map.querySourceFeatures(sourceKey, {
        sourceLayer: 'transportation'
    });
```
**Potential Error**: 
- `sourceKey` could be undefined, passed to `map.getSource(undefined)` - might throw
- `map.querySourceFeatures()` might return undefined if no features exist
- Line 157: `features.slice()` could fail if features is undefined

**Suggested Fix**:
```typescript
if (sourceKey && map.getSource(sourceKey)) {
    try {
        const features = map.querySourceFeatures(sourceKey, {
            sourceLayer: 'transportation'
        });
        if (features?.length) {
            console.log('🚗 Transportation features sample:', features.slice(0, 5));
        }
    } catch (error) {
        console.log('Unable to query features:', error);
    }
}
```

---

### 7. Unsafe Properties in Report Data  
**File**: `/home/user/floodsafe-mvp/apps/frontend/src/components/MapComponent.tsx`
**Lines**: 195-197
**Issue**: Accessing optional properties without type guards in HTML template
```typescript
const popupHTML = `
    ...
    <p><strong>Water Depth:</strong> <span class="capitalize">${props.water_depth}</span></p>
    <p><strong>Vehicle:</strong> <span class="capitalize">${props.vehicle_passability.replace('-', ' ')}</span></p>
    <p><strong>IoT Score:</strong> ${props.iot_validation_score}/100</p>
    ...
`;
```
**Potential Errors**:
- `props.water_depth` might be undefined or null → outputs "undefined"
- `props.vehicle_passability.replace()` will throw if `vehicle_passability` is undefined
- `props.iot_validation_score` could be missing

**Suggested Fix**:
```typescript
const popupHTML = `
    ...
    <p><strong>Water Depth:</strong> <span class="capitalize">${props.water_depth || 'unknown'}</span></p>
    <p><strong>Vehicle:</strong> <span class="capitalize">${(props.vehicle_passability || 'unknown').replace('-', ' ')}</span></p>
    <p><strong>IoT Score:</strong> ${props.iot_validation_score ?? 0}/100</p>
    ...
`;
```

---

## Moderate Issues

### 8. Inconsistent User Type Definitions
**File**: `/home/user/floodsafe-mvp/apps/frontend/src/components/screens/ProfileScreen.tsx`
**Line**: 24
**File**: `/home/user/floodsafe-mvp/apps/frontend/src/lib/api/hooks.ts`
**Line**: 44
**Issue**: Multiple definitions of `User` interface with different properties
```typescript
// ProfileScreen.tsx
interface User {
  id: string;
  username: string;
  email: string;
  phone?: string;
  profile_photo_url?: string;
  role: string;
  created_at: string;
  points: number;
  level: number;
  reports_count: number;
  verified_reports_count: number;
  badges?: string[];
  language?: string;
  notification_push?: boolean;
  notification_sms?: boolean;
  notification_whatsapp?: boolean;
  notification_email?: boolean;
  alert_preferences?: { watch: boolean; advisory: boolean; warning: boolean; emergency: boolean; };
}

// hooks.ts
export interface User {
    id: string;
    username: string;
    email: string;
    role: string;
    points: number;
    level: number;
    reports_count: number;
    verified_reports_count: number;
    badges: string[];
}
```
**Problem**: 
- ProfileScreen has more properties than hooks.ts version
- When API returns data, it might not include all expected properties
- Type mismatch between layers

**Suggested Fix**:
```typescript
// Create single definition in types.ts
export interface User {
    id: string;
    username: string;
    email: string;
    phone?: string;
    profile_photo_url?: string;
    role: string;
    created_at: string;
    points: number;
    level: number;
    reports_count: number;
    verified_reports_count: number;
    badges?: string[];
    language?: string;
    notification_push?: boolean;
    notification_sms?: boolean;
    notification_whatsapp?: boolean;
    notification_email?: boolean;
    alert_preferences?: {
        watch: boolean;
        advisory: boolean;
        warning: boolean;
        emergency: boolean;
    };
}

// Then import in both files:
import type { User } from '../types';
```

---

### 9. Missing Type Annotation in Map Callback
**File**: `/home/user/floodsafe-mvp/apps/frontend/src/components/MapComponent.tsx`
**Line**: 180
**Issue**: Event handler lacks full type annotation
```typescript
map.on('click', 'reports-layer', (e: maplibregl.MapMouseEvent) => {
    if (!e.features || e.features.length === 0) return;
    const feature = e.features[0];
```
**Problem**: 
- `feature` is inferred as `maplibregl.GeoJSONFeature` but not explicitly typed
- Could miss edge cases where feature structure is unexpected

**Suggested Fix**:
```typescript
map.on('click', 'reports-layer', (e: maplibregl.MapMouseEvent): void => {
    if (!e.features || e.features.length === 0) return;
    const feature: maplibregl.GeoJSONFeature = e.features[0];
    if (!feature.properties || !feature.geometry) return;
```

---

### 10. API Response Not Type-Guarded
**File**: `/home/user/floodsafe-mvp/apps/frontend/src/lib/api/hooks.ts`
**Line**: 166
**Issue**: Query invalidation without validating response shape
```typescript
queryClient.invalidateQueries({ queryKey: ['reports'] });
```
**Problem**: 
- No validation that returned data matches `Report[]` interface
- API might return different structure, causing silent failures downstream

**Suggested Fix**:
```typescript
// Add response validation
function validateReports(data: unknown): data is Report[] {
    return Array.isArray(data) && data.every(item => 
        'id' in item && 'description' in item && 'latitude' in item && 'longitude' in item
    );
}

export function useReportMutation() {
    const queryClient = useQueryClient();
    return useMutation({
        mutationFn: async (data: ReportCreate) => {
            // ... implementation
        },
        onSuccess: (data) => {
            if (validateReports(data)) {
                queryClient.invalidateQueries({ queryKey: ['reports'] });
            } else {
                console.error('Invalid report data returned from API');
            }
        },
    });
}
```

---

### 11. Missing User ID in Report Mutation
**File**: `/home/user/floodsafe-mvp/apps/frontend/src/lib/api/hooks.ts`
**Line**: 154-164
**Issue**: `user_id` is required but not provided
```typescript
export function useReportMutation() {
    return useMutation({
        mutationFn: async (data: ReportCreate) => {
            const formData = new FormData();
            formData.append('user_id', data.user_id); // ReportCreate has user_id
```
**Problem**:
- `ReportCreate` requires `user_id` but ReportScreen doesn't provide it
- See ReportScreen line 443: `await reportMutation.mutateAsync({ latitude, longitude, description, image: null })`
- This will cause API error: missing required field

**Suggested Fix**:
```typescript
// In ReportScreen.tsx, get user_id from auth context or state:
const handleSubmit = async () => {
    if (!location || !userId) { // Add userId check
        setErrorType('gps');
        setErrorMessage('User not authenticated. Please login first.');
        return;
    }
    
    await reportMutation.mutateAsync({
        user_id: userId,
        latitude: location.latitude,
        longitude: location.longitude,
        description: fullDescription,
        image: null
    });
}
```

---

### 12. LocationDetails Properties Might Be Undefined
**File**: `/home/user/floodsafe-mvp/apps/frontend/src/components/screens/HomeScreen.tsx`
**Lines**: 600-667
**Issue**: Accessing properties without null guards
```typescript
{locationDetails && (
    <div className="space-y-4">
        <div className="text-sm text-gray-600">
            <div className="flex items-center gap-2">
                <MapPin className="w-4 h-4" />
                <span>
                    {locationDetails.location.latitude.toFixed(4)}, {locationDetails.location.longitude.toFixed(4)}
                </span>
            </div>
            <div className="mt-1">
                Search Radius: {locationDetails.location.radius_meters}m
            </div>
        </div>
```
**Problem**: 
- Checking `locationDetails` exists but not nested properties like `location`
- Could throw "Cannot read property 'latitude' of undefined"

**Suggested Fix**:
```typescript
{locationDetails?.location && (
    <div className="text-sm text-gray-600">
        <div className="flex items-center gap-2">
            <MapPin className="w-4 h-4" />
            <span>
                {locationDetails.location.latitude?.toFixed(4) ?? 'N/A'}, 
                {locationDetails.location.longitude?.toFixed(4) ?? 'N/A'}
            </span>
        </div>
        <div className="mt-1">
            Search Radius: {locationDetails.location.radius_meters ?? 0}m
        </div>
    </div>
)}
```

---

## Summary Table

| Issue | File | Line(s) | Severity | Type |
|-------|------|---------|----------|------|
| Unsafe `as any` assertion | MapComponent.tsx | 54, 184 | Critical | Type Safety |
| Array filter/map chain without null check | HomeScreen.tsx | 73 | Critical | Null Safety |
| Direct array access without bounds check | HomeScreen.tsx | 88 | Critical | Array Safety |
| Missing type guard for event.results | ReportScreen.tsx | 112-118 | Critical | Type Safety |
| Unsafe window property access | ReportScreen.tsx | 88 | Critical | Type Safety |
| Missing optional chaining in map access | useMap.ts | 141-157 | Critical | Property Access |
| Unsafe report properties in HTML | MapComponent.tsx | 195-197 | Critical | Type Safety |
| Inconsistent User interface definitions | ProfileScreen.tsx, hooks.ts | 24, 44 | Moderate | Type Consistency |
| Missing type annotation in callback | MapComponent.tsx | 180 | Moderate | Type Annotation |
| No API response validation | hooks.ts | 166 | Moderate | Data Validation |
| Missing user_id in report mutation | hooks.ts, ReportScreen.tsx | 154-164, 443 | Moderate | Data Completeness |
| Undefined nested property access | HomeScreen.tsx | 600-667 | Moderate | Null Safety |

---

## Recommendations

1. **Enable stricter TypeScript settings** in `tsconfig.json`:
   - `"strict": true`
   - `"noImplicitAny": true`
   - `"strictNullChecks": true`
   - `"noUncheckedIndexedAccess": true`

2. **Create centralized type definitions** in `types.ts` for all interfaces used across files

3. **Add runtime validation** for API responses using a library like `zod` or `io-ts`

4. **Use type guards consistently** before accessing optional properties

5. **Remove all `as any` casts** and use proper types

6. **Add integration tests** to catch null/undefined errors at runtime

