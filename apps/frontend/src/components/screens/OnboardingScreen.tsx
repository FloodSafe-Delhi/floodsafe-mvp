/**
 * OnboardingScreen - 5-step wizard for new user setup
 *
 * Steps:
 * 1. City Selection (REQUIRED)
 * 2. Profile Info (username required, phone optional)
 * 3. Watch Areas (>=1 REQUIRED)
 * 4. Daily Routes (OPTIONAL)
 * 5. Completion (review & confirm)
 */

import { useReducer, useEffect, useState } from 'react';
import { useAuth } from '../../contexts/AuthContext';
import { useCityContext } from '../../contexts/CityContext';
import {
    useCreateWatchArea,
    useCreateDailyRoute,
    useUpdateUserOnboarding,
    useWatchAreas,
    useDailyRoutes,
    useGeocode
} from '../../lib/api/hooks';
import type { OnboardingFormState, OnboardingAction, WatchAreaCreate, DailyRouteCreate, GeocodingResult } from '../../types';
import type { CityKey } from '../../lib/map/cityConfigs';
import { CITIES } from '../../lib/map/cityConfigs';

// UI Components
import { Card } from '../ui/card';
import { Button } from '../ui/button';
import { Input } from '../ui/input';
import { Label } from '../ui/label';
import { Progress } from '../ui/progress';
import { RadioGroup, RadioGroupItem } from '../ui/radio-group';
import { MapPin, User, Bell, Route, CheckCircle, Trash2, ChevronLeft, ChevronRight, Loader2 } from 'lucide-react';
import { toast } from 'sonner';

// Initial state
const initialState: OnboardingFormState = {
    currentStep: 1,
    city: null,
    username: '',
    phone: '',
    watchAreas: [],
    dailyRoutes: [],
    errors: {},
    isSubmitting: false,
};

// Reducer
function onboardingReducer(state: OnboardingFormState, action: OnboardingAction): OnboardingFormState {
    switch (action.type) {
        case 'SET_CITY':
            return { ...state, city: action.payload, errors: {} };
        case 'SET_PROFILE':
            return { ...state, ...action.payload, errors: {} };
        case 'ADD_WATCH_AREA':
            return { ...state, watchAreas: [...state.watchAreas, action.payload] };
        case 'REMOVE_WATCH_AREA':
            return { ...state, watchAreas: state.watchAreas.filter((_, i) => i !== action.payload) };
        case 'ADD_DAILY_ROUTE':
            return { ...state, dailyRoutes: [...state.dailyRoutes, action.payload] };
        case 'REMOVE_DAILY_ROUTE':
            return { ...state, dailyRoutes: state.dailyRoutes.filter((_, i) => i !== action.payload) };
        case 'NEXT_STEP':
            return { ...state, currentStep: Math.min(state.currentStep + 1, 5) };
        case 'PREV_STEP':
            return { ...state, currentStep: Math.max(state.currentStep - 1, 1) };
        case 'SET_STEP':
            return { ...state, currentStep: action.payload };
        case 'SET_ERROR':
            return { ...state, errors: { ...state.errors, [action.payload.field]: action.payload.message } };
        case 'CLEAR_ERRORS':
            return { ...state, errors: {} };
        case 'SET_SUBMITTING':
            return { ...state, isSubmitting: action.payload };
        default:
            return state;
    }
}

interface OnboardingScreenProps {
    onComplete: () => void;
}

export function OnboardingScreen({ onComplete }: OnboardingScreenProps) {
    const { user } = useAuth();
    const { syncCityToUser } = useCityContext();
    const [state, dispatch] = useReducer(onboardingReducer, initialState);

    // Mutations
    const createWatchArea = useCreateWatchArea();
    const createDailyRoute = useCreateDailyRoute();
    const updateUser = useUpdateUserOnboarding();

    // Fetch existing data (for resume)
    const { data: existingWatchAreas = [] } = useWatchAreas(user?.id);
    const { data: existingDailyRoutes = [] } = useDailyRoutes(user?.id);

    // Initialize from user's saved progress
    useEffect(() => {
        if (user) {
            // Resume from saved step
            if (user.onboarding_step && user.onboarding_step > 1) {
                dispatch({ type: 'SET_STEP', payload: user.onboarding_step });
            }
            // Pre-fill city if already set
            if (user.city_preference) {
                dispatch({ type: 'SET_CITY', payload: user.city_preference as 'bangalore' | 'delhi' });
            }
            // Pre-fill username
            if (user.username) {
                dispatch({ type: 'SET_PROFILE', payload: { username: user.username, phone: user.phone || '' } });
            }
        }
    }, [user]);

    // Step validation
    const validateStep = (step: number): boolean => {
        dispatch({ type: 'CLEAR_ERRORS' });

        switch (step) {
            case 1:
                if (!state.city) {
                    dispatch({ type: 'SET_ERROR', payload: { field: 'city', message: 'Please select a city' } });
                    return false;
                }
                return true;
            case 2:
                if (!state.username || state.username.length < 3) {
                    dispatch({ type: 'SET_ERROR', payload: { field: 'username', message: 'Username must be at least 3 characters' } });
                    return false;
                }
                return true;
            case 3:
                // Check both local state and existing watch areas from backend
                const totalWatchAreas = state.watchAreas.length + existingWatchAreas.length;
                if (totalWatchAreas < 1) {
                    dispatch({ type: 'SET_ERROR', payload: { field: 'watchAreas', message: 'Add at least one watch area' } });
                    return false;
                }
                return true;
            case 4:
                // Optional step - always valid
                return true;
            case 5:
                // Completion step - always valid
                return true;
            default:
                return true;
        }
    };

    // Handle next step
    const handleNext = async () => {
        if (!validateStep(state.currentStep)) return;
        if (!user?.id) return;

        dispatch({ type: 'SET_SUBMITTING', payload: true });

        try {
            // Step-specific backend sync
            switch (state.currentStep) {
                case 1:
                    // Sync city to backend
                    if (state.city) {
                        await syncCityToUser(user.id, state.city);
                        await updateUser.mutateAsync({
                            userId: user.id,
                            data: { onboarding_step: 2, city_preference: state.city }
                        });
                    }
                    break;
                case 2:
                    // Sync profile to backend
                    await updateUser.mutateAsync({
                        userId: user.id,
                        data: {
                            onboarding_step: 3,
                            username: state.username,
                            phone: state.phone || undefined
                        }
                    });
                    break;
                case 3:
                    // Create watch areas
                    for (const wa of state.watchAreas) {
                        await createWatchArea.mutateAsync({ ...wa, user_id: user.id });
                    }
                    // Clear local state after successful save
                    state.watchAreas.forEach((_, i) => dispatch({ type: 'REMOVE_WATCH_AREA', payload: 0 }));
                    await updateUser.mutateAsync({ userId: user.id, data: { onboarding_step: 4 } });
                    break;
                case 4:
                    // Create daily routes (if any)
                    for (const route of state.dailyRoutes) {
                        await createDailyRoute.mutateAsync({ ...route, user_id: user.id });
                    }
                    // Clear local state after successful save
                    state.dailyRoutes.forEach((_, i) => dispatch({ type: 'REMOVE_DAILY_ROUTE', payload: 0 }));
                    await updateUser.mutateAsync({ userId: user.id, data: { onboarding_step: 5 } });
                    break;
                case 5:
                    // Mark onboarding complete
                    await updateUser.mutateAsync({ userId: user.id, data: { profile_complete: true } });
                    toast.success('Welcome to FloodSafe!');
                    onComplete();
                    return;
            }

            dispatch({ type: 'NEXT_STEP' });
        } catch (error) {
            console.error('Onboarding step failed:', error);
            toast.error('Something went wrong. Please try again.');
        } finally {
            dispatch({ type: 'SET_SUBMITTING', payload: false });
        }
    };

    // Handle skip (for optional steps)
    const handleSkip = async () => {
        if (!user?.id) return;

        dispatch({ type: 'SET_SUBMITTING', payload: true });
        try {
            await updateUser.mutateAsync({ userId: user.id, data: { onboarding_step: state.currentStep + 1 } });
            dispatch({ type: 'NEXT_STEP' });
        } catch (error) {
            console.error('Skip failed:', error);
            toast.error('Something went wrong. Please try again.');
        } finally {
            dispatch({ type: 'SET_SUBMITTING', payload: false });
        }
    };

    // Progress percentage
    const progress = (state.currentStep / 5) * 100;

    // Step titles
    const stepTitles = ['Select City', 'Your Profile', 'Watch Areas', 'Daily Routes', 'Complete'];
    const stepIcons = [MapPin, User, Bell, Route, CheckCircle];

    return (
        <div className="min-h-screen bg-gradient-to-br from-blue-50 to-cyan-50 p-4">
            <div className="max-w-lg mx-auto">
                {/* Header */}
                <div className="text-center mb-6">
                    <h1 className="text-2xl font-bold text-gray-900">Welcome to FloodSafe</h1>
                    <p className="text-gray-600 mt-1">Let's set up your account</p>
                </div>

                {/* Progress Bar */}
                <div className="mb-6">
                    <div className="flex justify-between mb-2">
                        {stepTitles.map((title, i) => {
                            const Icon = stepIcons[i];
                            const isActive = i + 1 === state.currentStep;
                            const isComplete = i + 1 < state.currentStep;
                            return (
                                <div
                                    key={i}
                                    className={`flex flex-col items-center ${isActive ? 'text-blue-600' : isComplete ? 'text-green-600' : 'text-gray-400'
                                        }`}
                                >
                                    <Icon className="w-5 h-5" />
                                    <span className="text-xs mt-1 hidden sm:block">{title}</span>
                                </div>
                            );
                        })}
                    </div>
                    <Progress value={progress} className="h-2" />
                    <p className="text-center text-sm text-gray-600 mt-2">
                        Step {state.currentStep} of 5: {stepTitles[state.currentStep - 1]}
                    </p>
                </div>

                {/* Step Content */}
                <Card className="p-6">
                    {state.currentStep === 1 && (
                        <Step1City
                            city={state.city}
                            onSelect={(city) => dispatch({ type: 'SET_CITY', payload: city })}
                            error={state.errors.city}
                        />
                    )}
                    {state.currentStep === 2 && (
                        <Step2Profile
                            username={state.username}
                            phone={state.phone}
                            onUpdate={(data) => dispatch({ type: 'SET_PROFILE', payload: data })}
                            errors={state.errors}
                        />
                    )}
                    {state.currentStep === 3 && (
                        <Step3WatchAreas
                            watchAreas={state.watchAreas}
                            existingWatchAreas={existingWatchAreas}
                            city={state.city}
                            onAdd={(wa) => dispatch({ type: 'ADD_WATCH_AREA', payload: wa })}
                            onRemove={(i) => dispatch({ type: 'REMOVE_WATCH_AREA', payload: i })}
                            error={state.errors.watchAreas}
                            userId={user?.id || ''}
                        />
                    )}
                    {state.currentStep === 4 && (
                        <Step4DailyRoutes
                            routes={state.dailyRoutes}
                            existingRoutes={existingDailyRoutes}
                            city={state.city}
                            onAdd={(route) => dispatch({ type: 'ADD_DAILY_ROUTE', payload: route })}
                            onRemove={(i) => dispatch({ type: 'REMOVE_DAILY_ROUTE', payload: i })}
                            userId={user?.id || ''}
                        />
                    )}
                    {state.currentStep === 5 && (
                        <Step5Completion
                            city={state.city}
                            username={state.username}
                            watchAreasCount={state.watchAreas.length + existingWatchAreas.length}
                            dailyRoutesCount={state.dailyRoutes.length + existingDailyRoutes.length}
                        />
                    )}
                </Card>

                {/* Navigation Buttons */}
                <div className="flex justify-between mt-6">
                    <Button
                        variant="outline"
                        onClick={() => dispatch({ type: 'PREV_STEP' })}
                        disabled={state.currentStep === 1 || state.isSubmitting}
                    >
                        <ChevronLeft className="w-4 h-4 mr-1" />
                        Back
                    </Button>

                    <div className="flex gap-2">
                        {/* Skip button for optional step 4 */}
                        {state.currentStep === 4 && (
                            <Button
                                variant="ghost"
                                onClick={handleSkip}
                                disabled={state.isSubmitting}
                            >
                                Skip
                            </Button>
                        )}

                        <Button
                            onClick={handleNext}
                            disabled={state.isSubmitting}
                        >
                            {state.isSubmitting ? (
                                <Loader2 className="w-4 h-4 mr-1 animate-spin" />
                            ) : null}
                            {state.currentStep === 5 ? 'Get Started' : 'Next'}
                            {state.currentStep < 5 && <ChevronRight className="w-4 h-4 ml-1" />}
                        </Button>
                    </div>
                </div>
            </div>
        </div>
    );
}

// ============================================================================
// STEP COMPONENTS
// ============================================================================

// Step 1: City Selection
interface Step1CityProps {
    city: 'bangalore' | 'delhi' | null;
    onSelect: (city: 'bangalore' | 'delhi') => void;
    error?: string;
}

function Step1City({ city, onSelect, error }: Step1CityProps) {
    return (
        <div className="space-y-4">
            <div>
                <h2 className="text-xl font-semibold mb-2">Select Your City</h2>
                <p className="text-gray-600 text-sm">
                    Choose the city where you want to receive flood alerts
                </p>
            </div>

            <RadioGroup value={city || ''} onValueChange={(v) => onSelect(v as 'bangalore' | 'delhi')}>
                {(['bangalore', 'delhi'] as const).map((cityKey) => (
                    <div
                        key={cityKey}
                        className={`flex items-center space-x-3 p-4 border rounded-lg cursor-pointer transition-colors ${city === cityKey ? 'border-blue-500 bg-blue-50' : 'border-gray-200 hover:border-gray-300'
                            }`}
                        onClick={() => onSelect(cityKey)}
                    >
                        <RadioGroupItem value={cityKey} id={cityKey} />
                        <Label htmlFor={cityKey} className="flex-1 cursor-pointer">
                            <div className="font-medium">{CITIES[cityKey].displayName}</div>
                            <div className="text-sm text-gray-500">
                                {cityKey === 'bangalore' ? 'Karnataka, India' : 'National Capital Territory, India'}
                            </div>
                        </Label>
                    </div>
                ))}
            </RadioGroup>

            {error && <p className="text-red-500 text-sm">{error}</p>}
        </div>
    );
}

// Step 2: Profile Info
interface Step2ProfileProps {
    username: string;
    phone: string;
    onUpdate: (data: { username: string; phone: string }) => void;
    errors: Record<string, string>;
}

function Step2Profile({ username, phone, onUpdate, errors }: Step2ProfileProps) {
    return (
        <div className="space-y-4">
            <div>
                <h2 className="text-xl font-semibold mb-2">Your Profile</h2>
                <p className="text-gray-600 text-sm">
                    Set up your profile information
                </p>
            </div>

            <div className="space-y-4">
                <div>
                    <Label htmlFor="username">Username *</Label>
                    <Input
                        id="username"
                        value={username}
                        onChange={(e) => onUpdate({ username: e.target.value, phone })}
                        placeholder="Enter your username"
                        className={errors.username ? 'border-red-500' : ''}
                    />
                    {errors.username && <p className="text-red-500 text-sm mt-1">{errors.username}</p>}
                </div>

                <div>
                    <Label htmlFor="phone">Phone Number (optional)</Label>
                    <Input
                        id="phone"
                        type="tel"
                        value={phone}
                        onChange={(e) => onUpdate({ username, phone: e.target.value })}
                        placeholder="+91 XXXXX XXXXX"
                    />
                    <p className="text-xs text-gray-500 mt-1">
                        Used for SMS alerts (optional)
                    </p>
                </div>
            </div>
        </div>
    );
}

// Step 3: Watch Areas
interface Step3WatchAreasProps {
    watchAreas: WatchAreaCreate[];
    existingWatchAreas: any[];
    city: 'bangalore' | 'delhi' | null;
    onAdd: (wa: WatchAreaCreate) => void;
    onRemove: (index: number) => void;
    error?: string;
    userId: string;
}

function Step3WatchAreas({ watchAreas, existingWatchAreas, city, onAdd, onRemove, error, userId }: Step3WatchAreasProps) {
    const [name, setName] = useState('');
    const [searchQuery, setSearchQuery] = useState('');

    // Use geocoding for location search
    const { data: searchResults = [] } = useGeocode(searchQuery, searchQuery.length >= 3);

    const handleAddFromSearch = (result: GeocodingResult) => {
        onAdd({
            user_id: userId,
            name: name || result.display_name.split(',')[0],
            latitude: parseFloat(result.lat),
            longitude: parseFloat(result.lon),
            radius: 1000,
        });
        setName('');
        setSearchQuery('');
    };

    const totalAreas = watchAreas.length + existingWatchAreas.length;

    return (
        <div className="space-y-4">
            <div>
                <h2 className="text-xl font-semibold mb-2">Watch Areas</h2>
                <p className="text-gray-600 text-sm">
                    Add locations you want to monitor for flood alerts (at least 1 required)
                </p>
            </div>

            {/* Add new watch area */}
            <div className="space-y-2">
                <Input
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    placeholder="Area name (e.g., Home, Office)"
                />
                <Input
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    placeholder="Search for a location..."
                />
                {searchResults.length > 0 && (
                    <div className="border rounded-lg divide-y max-h-40 overflow-y-auto">
                        {searchResults.map((result, i) => (
                            <button
                                key={i}
                                className="w-full p-2 text-left hover:bg-gray-50 text-sm"
                                onClick={() => handleAddFromSearch(result)}
                            >
                                {result.display_name}
                            </button>
                        ))}
                    </div>
                )}
            </div>

            {/* Existing watch areas from backend */}
            {existingWatchAreas.length > 0 && (
                <div className="space-y-2">
                    <Label className="text-sm text-gray-500">Previously added:</Label>
                    {existingWatchAreas.map((wa) => (
                        <div key={wa.id} className="flex items-center justify-between p-2 bg-green-50 rounded border border-green-200">
                            <div className="flex items-center gap-2">
                                <MapPin className="w-4 h-4 text-green-600" />
                                <span className="text-sm">{wa.name}</span>
                            </div>
                            <CheckCircle className="w-4 h-4 text-green-600" />
                        </div>
                    ))}
                </div>
            )}

            {/* New watch areas (pending save) */}
            {watchAreas.length > 0 && (
                <div className="space-y-2">
                    <Label className="text-sm text-gray-500">To be added:</Label>
                    {watchAreas.map((wa, i) => (
                        <div key={i} className="flex items-center justify-between p-2 bg-blue-50 rounded border border-blue-200">
                            <div className="flex items-center gap-2">
                                <MapPin className="w-4 h-4 text-blue-600" />
                                <span className="text-sm">{wa.name}</span>
                            </div>
                            <Button variant="ghost" size="sm" onClick={() => onRemove(i)}>
                                <Trash2 className="w-4 h-4 text-red-500" />
                            </Button>
                        </div>
                    ))}
                </div>
            )}

            {totalAreas === 0 && (
                <div className="text-center py-4 text-gray-500">
                    <MapPin className="w-8 h-8 mx-auto mb-2 opacity-50" />
                    <p className="text-sm">No watch areas added yet</p>
                </div>
            )}

            {error && <p className="text-red-500 text-sm">{error}</p>}
        </div>
    );
}

// Step 4: Daily Routes (simplified - showing structure)
interface Step4DailyRoutesProps {
    routes: DailyRouteCreate[];
    existingRoutes: any[];
    city: 'bangalore' | 'delhi' | null;
    onAdd: (route: DailyRouteCreate) => void;
    onRemove: (index: number) => void;
    userId: string;
}

function Step4DailyRoutes({ routes, existingRoutes }: Step4DailyRoutesProps) {
    return (
        <div className="space-y-4">
            <div>
                <h2 className="text-xl font-semibold mb-2">Daily Routes</h2>
                <p className="text-gray-600 text-sm">
                    Add your regular commute routes to get flood alerts along your path (optional)
                </p>
            </div>

            {/* Existing routes */}
            {existingRoutes.length > 0 && (
                <div className="space-y-2">
                    <Label className="text-sm text-gray-500">Your routes:</Label>
                    {existingRoutes.map((route) => (
                        <div key={route.id} className="flex items-center justify-between p-2 bg-green-50 rounded border border-green-200">
                            <div className="flex items-center gap-2">
                                <Route className="w-4 h-4 text-green-600" />
                                <span className="text-sm">{route.name}</span>
                                <span className="text-xs text-gray-500 capitalize">({route.transport_mode})</span>
                            </div>
                            <CheckCircle className="w-4 h-4 text-green-600" />
                        </div>
                    ))}
                </div>
            )}

            {/* New routes (pending) */}
            {routes.length > 0 && (
                <div className="space-y-2">
                    <Label className="text-sm text-gray-500">To be added:</Label>
                    {routes.map((route, i) => (
                        <div key={i} className="flex items-center justify-between p-2 bg-blue-50 rounded border border-blue-200">
                            <div className="flex items-center gap-2">
                                <Route className="w-4 h-4 text-blue-600" />
                                <span className="text-sm">{route.name}</span>
                            </div>
                            <Button variant="ghost" size="sm">
                                <Trash2 className="w-4 h-4 text-red-500" />
                            </Button>
                        </div>
                    ))}
                </div>
            )}

            {routes.length === 0 && existingRoutes.length === 0 && (
                <div className="text-center py-6 text-gray-500">
                    <Route className="w-8 h-8 mx-auto mb-2 opacity-50" />
                    <p className="text-sm">No daily routes added</p>
                    <p className="text-xs mt-1">You can add routes later from your profile</p>
                </div>
            )}

            <p className="text-sm text-gray-500 bg-gray-50 p-3 rounded">
                <strong>Tip:</strong> Add routes like "Home to Office" to receive alerts about flooding
                along your daily commute.
            </p>
        </div>
    );
}

// Step 5: Completion
interface Step5CompletionProps {
    city: 'bangalore' | 'delhi' | null;
    username: string;
    watchAreasCount: number;
    dailyRoutesCount: number;
}

function Step5Completion({ city, username, watchAreasCount, dailyRoutesCount }: Step5CompletionProps) {
    return (
        <div className="space-y-4 text-center">
            <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto">
                <CheckCircle className="w-10 h-10 text-green-600" />
            </div>

            <div>
                <h2 className="text-xl font-semibold mb-2">You're All Set!</h2>
                <p className="text-gray-600 text-sm">
                    Here's a summary of your setup:
                </p>
            </div>

            <div className="bg-gray-50 rounded-lg p-4 text-left space-y-3">
                <div className="flex justify-between">
                    <span className="text-gray-600">City:</span>
                    <span className="font-medium">{city ? CITIES[city].displayName : 'Not set'}</span>
                </div>
                <div className="flex justify-between">
                    <span className="text-gray-600">Username:</span>
                    <span className="font-medium">{username}</span>
                </div>
                <div className="flex justify-between">
                    <span className="text-gray-600">Watch Areas:</span>
                    <span className="font-medium">{watchAreasCount} area(s)</span>
                </div>
                <div className="flex justify-between">
                    <span className="text-gray-600">Daily Routes:</span>
                    <span className="font-medium">{dailyRoutesCount} route(s)</span>
                </div>
            </div>

            <p className="text-sm text-gray-500">
                Click "Get Started" to begin using FloodSafe
            </p>
        </div>
    );
}
