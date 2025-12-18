/**
 * Login Screen for FloodSafe.
 *
 * Professional UI with responsive design for desktop and mobile.
 * Primary: Google Sign-In | Secondary: Phone OTP
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import { useAuth } from '../../contexts/AuthContext';
import { AlertCircle, Loader2, Shield, MapPin, Bell, Phone, ArrowRight, ArrowLeft, Check, FileEdit, Mail, Eye, EyeOff } from 'lucide-react';

const GOOGLE_CLIENT_ID = import.meta.env.VITE_GOOGLE_CLIENT_ID || '';

interface LoginScreenProps {
    onLoginSuccess?: () => void;
}

declare global {
    interface Window {
        google?: {
            accounts: {
                id: {
                    initialize: (config: {
                        client_id: string;
                        callback: (response: { credential: string }) => void;
                        auto_select?: boolean;
                        context?: string;
                    }) => void;
                    renderButton: (
                        element: HTMLElement,
                        options: {
                            theme?: 'outline' | 'filled_blue' | 'filled_black';
                            size?: 'large' | 'medium' | 'small';
                            width?: number;
                            text?: 'signin_with' | 'signin' | 'continue_with' | 'signup_with';
                            shape?: 'rectangular' | 'pill' | 'circle' | 'square';
                            logo_alignment?: 'left' | 'center';
                        }
                    ) => void;
                    prompt: () => void;
                };
            };
        };
    }
}

export function LoginScreen({ onLoginSuccess }: LoginScreenProps) {
    const { loginWithGoogle, registerWithEmail, loginWithEmail, isLoading, error, clearError } = useAuth();

    const [authMethod, setAuthMethod] = useState<'email' | 'google' | 'phone'>('email');
    const [activeFeature, setActiveFeature] = useState<'report' | 'routes' | 'alerts'>('report');
    const [localError, setLocalError] = useState<string | null>(null);
    const [scriptStatus, setScriptStatus] = useState<'loading' | 'ready' | 'error'>('loading');
    const googleButtonRef = useRef<HTMLDivElement>(null);
    const initAttempted = useRef(false);

    // Email state
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [showPassword, setShowPassword] = useState(false);
    const [isSignUp, setIsSignUp] = useState(false);

    // Phone state
    const [phoneNumber, setPhoneNumber] = useState('');
    const [countryCode, setCountryCode] = useState('+91');
    const [otpStep, setOtpStep] = useState(false);
    const [otp, setOtp] = useState(['', '', '', '', '', '']);
    const [countdown, setCountdown] = useState(0);
    const otpRefs = useRef<(HTMLInputElement | null)[]>([]);

    useEffect(() => {
        clearError();
        setLocalError(null);
    }, [clearError, authMethod]);

    const handleGoogleCallback = useCallback(async (response: { credential: string }) => {
        try {
            setLocalError(null);
            await loginWithGoogle(response.credential);
            onLoginSuccess?.();
        } catch (err) {
            setLocalError(err instanceof Error ? err.message : 'Google sign-in failed');
        }
    }, [loginWithGoogle, onLoginSuccess]);

    const initializeGoogleSignIn = useCallback(() => {
        if (!window.google || !googleButtonRef.current || initAttempted.current) return;
        initAttempted.current = true;

        try {
            window.google.accounts.id.initialize({
                client_id: GOOGLE_CLIENT_ID,
                callback: handleGoogleCallback,
                auto_select: false,
                context: 'signin',
            });

            googleButtonRef.current.innerHTML = '';
            window.google.accounts.id.renderButton(googleButtonRef.current, {
                theme: 'outline',
                size: 'large',
                width: 260,
                text: 'signin_with',
                shape: 'rectangular',
                logo_alignment: 'left',
            });
        } catch (err) {
            setScriptStatus('error');
            setLocalError('Failed to initialize Google Sign-In');
        }
    }, [handleGoogleCallback]);

    useEffect(() => {
        if (!GOOGLE_CLIENT_ID) {
            setScriptStatus('error');
            setLocalError('Google Sign-In is not configured');
            return;
        }

        if (window.google?.accounts?.id) {
            setScriptStatus('ready');
            return;
        }

        const existingScript = document.querySelector('script[src="https://accounts.google.com/gsi/client"]');
        if (existingScript) {
            const checkGoogle = setInterval(() => {
                if (window.google?.accounts?.id) {
                    clearInterval(checkGoogle);
                    setScriptStatus('ready');
                }
            }, 100);
            setTimeout(() => {
                clearInterval(checkGoogle);
                if (!window.google?.accounts?.id) {
                    setScriptStatus('error');
                }
            }, 10000);
            return;
        }

        const script = document.createElement('script');
        script.src = 'https://accounts.google.com/gsi/client';
        script.async = true;
        script.defer = true;
        script.onload = () => {
            const checkGoogle = setInterval(() => {
                if (window.google?.accounts?.id) {
                    clearInterval(checkGoogle);
                    setScriptStatus('ready');
                }
            }, 50);
            setTimeout(() => {
                clearInterval(checkGoogle);
                if (!window.google?.accounts?.id) setScriptStatus('error');
            }, 5000);
        };
        script.onerror = () => setScriptStatus('error');
        document.head.appendChild(script);
    }, []);

    useEffect(() => {
        if (scriptStatus === 'ready' && googleButtonRef.current && !initAttempted.current) {
            initializeGoogleSignIn();
        }
    }, [scriptStatus, initializeGoogleSignIn]);

    useEffect(() => {
        if (authMethod === 'google' && scriptStatus === 'ready' && googleButtonRef.current) {
            const timer = setTimeout(() => {
                if (googleButtonRef.current && window.google?.accounts?.id) {
                    try {
                        // Must call initialize() before renderButton()
                        window.google.accounts.id.initialize({
                            client_id: GOOGLE_CLIENT_ID,
                            callback: handleGoogleCallback,
                            auto_select: false,
                            context: 'signin',
                        });

                        googleButtonRef.current.innerHTML = '';
                        window.google.accounts.id.renderButton(googleButtonRef.current, {
                            theme: 'outline',
                            size: 'large',
                            width: 260,
                            text: 'signin_with',
                            shape: 'rectangular',
                            logo_alignment: 'left',
                        });
                    } catch (err) {
                        console.error('Google Sign-In render error:', err);
                    }
                }
            }, 100);
            return () => clearTimeout(timer);
        }
    }, [authMethod, scriptStatus, handleGoogleCallback]);

    useEffect(() => {
        if (countdown > 0) {
            const timer = setTimeout(() => setCountdown(countdown - 1), 1000);
            return () => clearTimeout(timer);
        }
    }, [countdown]);

    // Email form handler
    const handleEmailSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setLocalError(null);

        // Basic validation
        if (!email || !email.includes('@')) {
            setLocalError('Please enter a valid email address');
            return;
        }
        if (password.length < 8) {
            setLocalError('Password must be at least 8 characters');
            return;
        }

        try {
            if (isSignUp) {
                await registerWithEmail(email, password);
            } else {
                await loginWithEmail(email, password);
            }
            onLoginSuccess?.();
        } catch (err) {
            // Error is already set in context, but we can add local handling
            if (err instanceof Error) {
                setLocalError(err.message);
            }
        }
    };

    const handlePhoneSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        if (phoneNumber.length >= 10) {
            setOtpStep(true);
            setCountdown(30);
            setTimeout(() => otpRefs.current[0]?.focus(), 100);
        }
    };

    const handleOtpChange = (index: number, value: string) => {
        const digit = value.replace(/[^0-9]/g, '').slice(-1);
        const newOtp = [...otp];
        newOtp[index] = digit;
        setOtp(newOtp);
        if (digit && index < 5) otpRefs.current[index + 1]?.focus();
    };

    const handleOtpKeyDown = (index: number, e: React.KeyboardEvent) => {
        if (e.key === 'Backspace' && !otp[index] && index > 0) {
            otpRefs.current[index - 1]?.focus();
        }
    };

    const handleOtpPaste = (e: React.ClipboardEvent) => {
        e.preventDefault();
        const paste = e.clipboardData.getData('text').replace(/[^0-9]/g, '').slice(0, 6);
        const newOtp = [...otp];
        paste.split('').forEach((digit, i) => { newOtp[i] = digit; });
        setOtp(newOtp);
        if (paste.length > 0) otpRefs.current[Math.min(paste.length, 5)]?.focus();
    };

    const isOtpComplete = otp.every(d => d !== '');
    const displayError = localError || error;

    const features = [
        { id: 'report', label: 'Report', icon: FileEdit },
        { id: 'routes', label: 'Routes', icon: MapPin },
        { id: 'alerts', label: 'Alerts', icon: Bell },
    ] as const;

    return (
        <div className="min-h-screen w-full flex items-center justify-center p-4 sm:p-6 lg:p-8"
             style={{ background: 'linear-gradient(145deg, #eef4ff 0%, #e0ecff 50%, #f0f6ff 100%)' }}>

            {/* Subtle dot pattern */}
            <div className="fixed inset-0 pointer-events-none opacity-30"
                 style={{
                     backgroundImage: 'radial-gradient(circle at 1px 1px, #94b8ed 0.5px, transparent 0.5px)',
                     backgroundSize: '24px 24px'
                 }} />

            {/* Main Card - professional centered card */}
            <div className="relative w-full" style={{ maxWidth: '460px' }}>
                <div className="bg-white overflow-hidden border border-gray-100/50 shadow-2xl" style={{ borderRadius: '32px' }}>

                    {/* Top accent bar */}
                    <div className="h-1 bg-gradient-to-r from-blue-600 via-blue-500 to-cyan-400" />

                    {/* Content */}
                    <div className="px-10 pt-12 pb-10 sm:px-14 sm:pt-14 sm:pb-12">

                        {/* Logo */}
                        <div className="text-center mb-10">
                            <div className="mx-auto w-16 h-16 mb-5 text-blue-600">
                                <svg viewBox="0 0 48 48" fill="none" className="w-full h-full">
                                    <path d="M24 4L6 12V24C6 36 14 44 24 48C34 44 42 36 42 24V12L24 4Z"
                                          stroke="currentColor" strokeWidth="2.5" fill="none" strokeLinecap="round" strokeLinejoin="round"/>
                                    <ellipse cx="24" cy="26" rx="6" ry="4" stroke="currentColor" strokeWidth="2" fill="none"/>
                                    <circle cx="24" cy="26" r="2" fill="currentColor"/>
                                </svg>
                            </div>
                            <h1 className="text-2xl sm:text-3xl font-bold text-gray-900 tracking-tight">
                                FloodSafe
                            </h1>
                            <p className="text-gray-500 text-sm mt-2">Community flood monitoring</p>
                        </div>

                        {/* Feature tabs - centered pill buttons */}
                        <div className="flex justify-center gap-3 mt-2 mb-12">
                            {features.map(({ id, label, icon: Icon }) => (
                                <button
                                    key={id}
                                    onClick={() => setActiveFeature(id)}
                                    className={`flex items-center justify-center gap-2 px-6 py-3 rounded-full text-sm font-medium transition-all duration-200 ${
                                        activeFeature === id
                                            ? 'bg-blue-600 text-white shadow-md'
                                            : 'text-gray-600 hover:bg-gray-100'
                                    }`}
                                    style={activeFeature !== id ? { backgroundColor: '#f3f4f6' } : undefined}
                                >
                                    <Icon className="w-4 h-4" />
                                    {label}
                                </button>
                            ))}
                        </div>

                        {/* Sign in prompt */}
                        <p className="text-center text-gray-500 text-sm mb-6">
                            Sign in to get started
                        </p>

                        {/* Auth toggle - centered pill buttons */}
                        <div className="flex justify-center gap-2 mb-6">
                            <button
                                onClick={() => setAuthMethod('email')}
                                className={`flex items-center justify-center gap-2 px-5 py-2.5 rounded-full text-sm font-medium transition-all duration-200 ${
                                    authMethod === 'email'
                                        ? 'bg-blue-600 text-white shadow-md'
                                        : 'text-gray-600 hover:bg-gray-200'
                                }`}
                                style={authMethod !== 'email' ? { backgroundColor: '#f3f4f6' } : undefined}
                            >
                                <Mail className="w-4 h-4" />
                                Email
                            </button>
                            <button
                                onClick={() => setAuthMethod('google')}
                                className={`flex items-center justify-center gap-2 px-5 py-2.5 rounded-full text-sm font-medium transition-all duration-200 ${
                                    authMethod === 'google'
                                        ? 'bg-blue-600 text-white shadow-md'
                                        : 'text-gray-600 hover:bg-gray-200'
                                }`}
                                style={authMethod !== 'google' ? { backgroundColor: '#f3f4f6' } : undefined}
                            >
                                <svg viewBox="0 0 24 24" className="w-4 h-4">
                                    <path fill={authMethod === 'google' ? '#fff' : '#4285F4'} d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
                                    <path fill={authMethod === 'google' ? '#fff' : '#34A853'} d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
                                    <path fill={authMethod === 'google' ? '#fff' : '#FBBC05'} d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
                                    <path fill={authMethod === 'google' ? '#fff' : '#EA4335'} d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
                                </svg>
                                Google
                            </button>
                            <button
                                onClick={() => setAuthMethod('phone')}
                                className={`flex items-center justify-center gap-2 px-5 py-2.5 rounded-full text-sm font-medium transition-all duration-200 ${
                                    authMethod === 'phone'
                                        ? 'bg-blue-600 text-white shadow-md'
                                        : 'text-gray-600 hover:bg-gray-200'
                                }`}
                                style={authMethod !== 'phone' ? { backgroundColor: '#f3f4f6' } : undefined}
                            >
                                <Phone className="w-4 h-4" />
                                Phone
                            </button>
                        </div>

                        {/* Error */}
                        {displayError && (
                            <div className="mb-5 p-3 bg-red-50 border border-red-100 rounded-xl flex items-start gap-2">
                                <AlertCircle className="w-4 h-4 text-red-500 flex-shrink-0 mt-0.5" />
                                <p className="text-sm text-red-600">{displayError}</p>
                            </div>
                        )}

                        {/* Email Panel */}
                        {authMethod === 'email' && (
                            <div className="mt-2">
                                <form onSubmit={handleEmailSubmit} className="space-y-4">
                                    {/* Sign In / Sign Up toggle */}
                                    <div className="flex justify-center mb-4">
                                        <div className="inline-flex bg-gray-100 rounded-lg p-1">
                                            <button
                                                type="button"
                                                onClick={() => setIsSignUp(false)}
                                                className={`px-4 py-2 text-sm font-medium rounded-md transition-all ${
                                                    !isSignUp
                                                        ? 'bg-white text-gray-900 shadow-sm'
                                                        : 'text-gray-500 hover:text-gray-700'
                                                }`}
                                            >
                                                Sign In
                                            </button>
                                            <button
                                                type="button"
                                                onClick={() => setIsSignUp(true)}
                                                className={`px-4 py-2 text-sm font-medium rounded-md transition-all ${
                                                    isSignUp
                                                        ? 'bg-white text-gray-900 shadow-sm'
                                                        : 'text-gray-500 hover:text-gray-700'
                                                }`}
                                            >
                                                Sign Up
                                            </button>
                                        </div>
                                    </div>

                                    {/* Email input */}
                                    <div>
                                        <label htmlFor="email" className="block text-sm font-medium text-gray-700 mb-1.5">
                                            Email address
                                        </label>
                                        <input
                                            id="email"
                                            type="email"
                                            value={email}
                                            onChange={(e) => setEmail(e.target.value)}
                                            placeholder="you@example.com"
                                            className="w-full px-4 py-3 border-2 border-gray-200 rounded-xl text-gray-900 placeholder-gray-400 focus:outline-none focus:border-blue-500 focus:ring-4 focus:ring-blue-50 transition-all"
                                            autoComplete="email"
                                        />
                                    </div>

                                    {/* Password input */}
                                    <div>
                                        <label htmlFor="password" className="block text-sm font-medium text-gray-700 mb-1.5">
                                            Password
                                        </label>
                                        <div className="relative">
                                            <input
                                                id="password"
                                                type={showPassword ? 'text' : 'password'}
                                                value={password}
                                                onChange={(e) => setPassword(e.target.value)}
                                                placeholder={isSignUp ? 'Min 8 characters' : 'Your password'}
                                                className="w-full px-4 py-3 pr-12 border-2 border-gray-200 rounded-xl text-gray-900 placeholder-gray-400 focus:outline-none focus:border-blue-500 focus:ring-4 focus:ring-blue-50 transition-all"
                                                autoComplete={isSignUp ? 'new-password' : 'current-password'}
                                            />
                                            <button
                                                type="button"
                                                onClick={() => setShowPassword(!showPassword)}
                                                className="absolute right-3 top-1/2 -translate-y-1/2 p-1.5 text-gray-400 hover:text-gray-600 transition-colors"
                                            >
                                                {showPassword ? (
                                                    <EyeOff className="w-5 h-5" />
                                                ) : (
                                                    <Eye className="w-5 h-5" />
                                                )}
                                            </button>
                                        </div>
                                        {isSignUp && (
                                            <p className="text-xs text-gray-500 mt-1.5">
                                                Password must be at least 8 characters
                                            </p>
                                        )}
                                    </div>

                                    {/* Submit button */}
                                    <button
                                        type="submit"
                                        disabled={isLoading || !email || password.length < 8}
                                        className="w-full py-3.5 bg-blue-600 hover:bg-blue-700 disabled:opacity-40 disabled:cursor-not-allowed text-white font-medium rounded-xl flex items-center justify-center gap-2 transition-all shadow-md shadow-blue-500/20"
                                    >
                                        {isLoading ? (
                                            <>
                                                <Loader2 className="w-4 h-4 animate-spin" />
                                                {isSignUp ? 'Creating account...' : 'Signing in...'}
                                            </>
                                        ) : (
                                            <>
                                                {isSignUp ? 'Create Account' : 'Sign In'}
                                                <ArrowRight className="w-4 h-4" />
                                            </>
                                        )}
                                    </button>
                                </form>
                            </div>
                        )}

                        {/* Google Panel */}
                        {authMethod === 'google' && (
                            <div className="space-y-4">
                                {scriptStatus === 'loading' && (
                                    <div className="h-11 bg-gray-100 rounded-lg animate-pulse" />
                                )}
                                {scriptStatus === 'error' && !displayError && (
                                    <div className="text-center p-4 bg-amber-50 rounded-xl border border-amber-100">
                                        <p className="text-sm text-amber-700">
                                            Unable to load Google Sign-In.
                                            <button onClick={() => window.location.reload()} className="block mx-auto mt-1 text-amber-600 underline text-xs">
                                                Refresh to retry
                                            </button>
                                        </p>
                                    </div>
                                )}
                                {scriptStatus === 'ready' && (
                                    <div ref={googleButtonRef} className="flex justify-center" />
                                )}
                                {isLoading && (
                                    <div className="flex items-center justify-center gap-2 text-gray-500 py-2">
                                        <Loader2 className="w-4 h-4 animate-spin" />
                                        <span className="text-sm">Signing in...</span>
                                    </div>
                                )}
                            </div>
                        )}

                        {/* Phone Panel */}
                        {authMethod === 'phone' && (
                            <div className="mt-2">
                                {!otpStep ? (
                                    <form onSubmit={handlePhoneSubmit} className="space-y-6">
                                        {/* Phone input - centered with max width */}
                                        <div className="max-w-xs mx-auto">
                                            <div className="flex border-2 border-gray-200 rounded-2xl overflow-hidden focus-within:border-blue-500 focus-within:ring-4 focus-within:ring-blue-50 transition-all bg-white">
                                                <select
                                                    value={countryCode}
                                                    onChange={(e) => setCountryCode(e.target.value)}
                                                    className="px-3 py-4 bg-gray-50 border-r border-gray-200 text-gray-700 text-sm focus:outline-none font-medium"
                                                >
                                                    <option value="+91">🇮🇳 +91</option>
                                                    <option value="+1">🇺🇸 +1</option>
                                                    <option value="+44">🇬🇧 +44</option>
                                                    <option value="+61">🇦🇺 +61</option>
                                                </select>
                                                <input
                                                    type="tel"
                                                    value={phoneNumber}
                                                    onChange={(e) => setPhoneNumber(e.target.value.replace(/[^0-9]/g, ''))}
                                                    placeholder="Phone number"
                                                    className="flex-1 px-4 py-4 text-gray-900 placeholder-gray-400 focus:outline-none text-base"
                                                    maxLength={10}
                                                />
                                            </div>
                                        </div>
                                        {/* Send code button - centered, not full width */}
                                        <div className="flex justify-center">
                                            <button
                                                type="submit"
                                                disabled={phoneNumber.length < 10}
                                                className="px-16 py-3.5 bg-blue-600 hover:bg-blue-700 disabled:opacity-40 disabled:cursor-not-allowed text-white font-medium rounded-2xl flex items-center justify-center gap-2 transition-all shadow-md shadow-blue-500/20"
                                            >
                                                Send code
                                                <ArrowRight className="w-4 h-4" />
                                            </button>
                                        </div>
                                    </form>
                                ) : (
                                    <div className="space-y-4">
                                        <button
                                            onClick={() => { setOtpStep(false); setOtp(['', '', '', '', '', '']); }}
                                            className="flex items-center gap-1.5 text-gray-500 hover:text-blue-600 text-sm transition-colors"
                                        >
                                            <ArrowLeft className="w-4 h-4" />
                                            Change number
                                        </button>
                                        <p className="text-sm text-gray-600">
                                            Code sent to <span className="font-medium text-gray-900">{countryCode} {phoneNumber}</span>
                                        </p>
                                        <div className="flex gap-2 justify-center" onPaste={handleOtpPaste}>
                                            {otp.map((digit, index) => (
                                                <input
                                                    key={index}
                                                    ref={(el) => { otpRefs.current[index] = el; }}
                                                    type="text"
                                                    inputMode="numeric"
                                                    maxLength={1}
                                                    value={digit}
                                                    onChange={(e) => handleOtpChange(index, e.target.value)}
                                                    onKeyDown={(e) => handleOtpKeyDown(index, e)}
                                                    className={`w-10 h-12 sm:w-11 sm:h-13 text-center text-lg font-semibold border rounded-lg transition-all focus:outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-100 ${
                                                        digit ? 'border-blue-500 bg-blue-50' : 'border-gray-200 bg-gray-50'
                                                    }`}
                                                />
                                            ))}
                                        </div>
                                        <p className="text-xs text-gray-500 text-center">
                                            {countdown > 0 ? `Resend in ${countdown}s` : (
                                                <button onClick={() => setCountdown(30)} className="text-blue-600 hover:underline">
                                                    Resend code
                                                </button>
                                            )}
                                        </p>
                                        <button
                                            disabled={!isOtpComplete}
                                            className="w-full py-3.5 bg-blue-600 hover:bg-blue-700 disabled:opacity-40 disabled:cursor-not-allowed text-white font-medium rounded-xl flex items-center justify-center gap-2 transition-all"
                                        >
                                            Verify
                                            <Check className="w-4 h-4" />
                                        </button>
                                    </div>
                                )}
                                <div id="recaptcha-container" />
                            </div>
                        )}

                        {/* Terms */}
                        <p className="text-center text-xs text-gray-400 leading-relaxed mt-8 mb-4">
                            By continuing, you agree to our{' '}
                            <span className="text-blue-600 hover:underline cursor-pointer">Terms</span>
                            {' '}and{' '}
                            <span className="text-blue-600 hover:underline cursor-pointer">Privacy Policy</span>
                        </p>

                        {/* Security badge */}
                        <div className="flex items-center justify-center gap-2 py-2.5 text-gray-400 text-xs">
                            <Shield className="w-3.5 h-3.5" />
                            Secure authentication
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}

export default LoginScreen;
