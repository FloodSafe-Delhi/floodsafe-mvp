/**
 * Login Screen for FloodSafe.
 *
 * "Civic Authority" design: Full-width dark brand bar at top for strong brand
 * presence, photo as contextual atmosphere (not hero), clean form below.
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import { useAuth } from '../../contexts/AuthContext';
import { useLanguage, type AppLanguage } from '../../contexts/LanguageContext';
import { t } from '../../lib/onboarding-bot/translations';
import { cn } from '../../lib/utils';
import { AlertCircle, Loader2, Shield, Phone, ArrowRight, ArrowLeft, Eye, EyeOff, Globe } from 'lucide-react';
import { API_BASE_URL } from '../../lib/api/config';
import { TokenStorage } from '../../lib/auth/token-storage';

const GOOGLE_CLIENT_ID = import.meta.env.VITE_GOOGLE_CLIENT_ID || '';

function PasswordStrengthIndicator({ password }: { password: string }) {
    const rules = [
        { label: 'At least 8 characters', met: password.length >= 8 },
        { label: 'Uppercase letter', met: /[A-Z]/.test(password) },
        { label: 'Lowercase letter', met: /[a-z]/.test(password) },
        { label: 'Number', met: /\d/.test(password) },
        { label: 'Special character', met: /[^A-Za-z0-9]/.test(password) },
    ];
    if (!password) return null;
    return (
        <ul className="text-xs space-y-0.5 mt-1.5">
            {rules.map((r) => (
                <li key={r.label} className={r.met ? 'text-green-600' : 'text-gray-400'}>
                    {r.met ? '✓' : '○'} {r.label}
                </li>
            ))}
        </ul>
    );
}

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
    const { language, setLanguage } = useLanguage();

    const LANG_OPTIONS: { code: AppLanguage; label: string }[] = [
        { code: 'en', label: 'EN' },
        { code: 'hi', label: 'हिंदी' },
        { code: 'id', label: 'Bahasa' },
    ];

    const [authMethod, setAuthMethod] = useState<'email' | 'phone'>('email');
    const [localError, setLocalError] = useState<string | null>(null);
    const [lockedUntil, setLockedUntil] = useState<Date | null>(null);
    const [lockoutMinutes, setLockoutMinutes] = useState(0);
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

    // WhatsApp login state
    const [loginCode, setLoginCode] = useState('');
    const [sessionId, setSessionId] = useState('');
    const [waLink, setWaLink] = useState<string | null>(null);
    const [expiresAt, setExpiresAt] = useState<Date | null>(null);
    const [isPhoneLoading, setIsPhoneLoading] = useState(false);

    useEffect(() => {
        clearError();
        setLocalError(null);
    }, [clearError, authMethod]);

    // ── Google Sign-In setup ──
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
                width: 300,
                text: 'signin_with',
                shape: 'rectangular',
                logo_alignment: 'left',
            });
        } catch {
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
                if (window.google?.accounts?.id) { clearInterval(checkGoogle); setScriptStatus('ready'); }
            }, 100);
            setTimeout(() => { clearInterval(checkGoogle); if (!window.google?.accounts?.id) setScriptStatus('error'); }, 10000);
            return;
        }
        const script = document.createElement('script');
        script.src = 'https://accounts.google.com/gsi/client';
        script.async = true;
        script.defer = true;
        script.onload = () => {
            const checkGoogle = setInterval(() => {
                if (window.google?.accounts?.id) { clearInterval(checkGoogle); setScriptStatus('ready'); }
            }, 50);
            setTimeout(() => { clearInterval(checkGoogle); if (!window.google?.accounts?.id) setScriptStatus('error'); }, 5000);
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
        if (authMethod === 'email' && scriptStatus === 'ready' && googleButtonRef.current) {
            const timer = setTimeout(() => {
                if (googleButtonRef.current && window.google?.accounts?.id) {
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
                            width: 300,
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

    // ── WhatsApp login polling ──
    useEffect(() => {
        if (!sessionId || !otpStep) return;

        const interval = setInterval(async () => {
            try {
                const response = await fetch(
                    `${API_BASE_URL}/auth/whatsapp-login/status?session_id=${sessionId}`
                );
                const data = await response.json();

                if (data.status === 'verified' && data.access_token) {
                    clearInterval(interval);
                    TokenStorage.setTokens(data.access_token, data.refresh_token);
                    window.location.href = '/app';
                } else if (data.status === 'expired') {
                    clearInterval(interval);
                    setLocalError('Verification timed out. Please try again.');
                    setOtpStep(false);
                    setLoginCode('');
                    setSessionId('');
                    setWaLink(null);
                    setExpiresAt(null);
                }
            } catch {
                // Network error — keep polling
            }
        }, 2000);

        return () => clearInterval(interval);
    }, [sessionId, otpStep]);

    // ── Form handlers ──
    const handleEmailSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setLocalError(null);
        if (!email || !email.includes('@')) { setLocalError(t(language, 'login.error.email')); return; }
        if (password.length < 8) { setLocalError(t(language, 'login.error.password')); return; }
        try {
            if (isSignUp) { await registerWithEmail(email, password); }
            else { await loginWithEmail(email, password); }
            onLoginSuccess?.();
        } catch (err: unknown) {
            if (err instanceof Error) {
                try {
                    const parsed = JSON.parse(err.message);
                    if (parsed.locked_until) {
                        setLockedUntil(new Date(parsed.locked_until));
                        setLockoutMinutes(parsed.remaining_minutes || 15);
                        return;
                    }
                } catch { /* not JSON — fall through to regular error */ }
                setLocalError(err.message);
            }
        }
    };

    const handlePhoneSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (phoneNumber.length < 8) return;
        setLocalError(null);
        setIsPhoneLoading(true);
        try {
            const countryMap: Record<string, string> = { '+91': 'IN', '+62': 'ID', '+65': 'SG' };
            const response = await fetch(`${API_BASE_URL}/auth/whatsapp-login/initiate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    phone: phoneNumber,
                    country_code: countryMap[countryCode] || 'IN',
                }),
            });
            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || 'Failed to initiate login');
            }
            const data = await response.json();
            setLoginCode(data.code);
            setSessionId(data.session_id);
            setWaLink(data.wa_link);
            setExpiresAt(new Date(data.expires_at));
            setOtpStep(true);
        } catch (err: unknown) {
            setLocalError(err instanceof Error ? err.message : 'Failed to start WhatsApp login');
        } finally {
            setIsPhoneLoading(false);
        }
    };

    const displayError = localError || error;

    return (
        <div className="min-h-screen w-full flex flex-col bg-background">

            {/* Language selector */}
            <div className="flex items-center justify-center gap-1 py-2 bg-white/80 backdrop-blur-sm">
                <Globe className="w-4 h-4 text-muted-foreground" />
                {LANG_OPTIONS.map(({ code, label }) => (
                    <button
                        key={code}
                        onClick={() => setLanguage(code)}
                        className={cn(
                            'px-3 py-1 rounded-full text-xs font-medium transition-colors',
                            language === code
                                ? 'bg-blue-600 text-white'
                                : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                        )}
                    >
                        {label}
                    </button>
                ))}
            </div>

            {/* ════════════════════════════════════════════════════
                BRAND BAR — Full width, top of entire page
                Dark navy bg, white text. This IS the brand presence.
                ════════════════════════════════════════════════════ */}
            <header className="bg-primary text-primary-foreground shrink-0">
                <div className="max-w-5xl mx-auto px-5 py-4 flex items-center justify-between">
                    <div className="flex items-center gap-3">
                        <div className="w-9 h-9 rounded-lg border border-primary-foreground/20 flex items-center justify-center">
                            <Shield className="w-5 h-5" />
                        </div>
                        <div>
                            <h1 className="text-base font-bold tracking-tight leading-none">FloodSafe</h1>
                            <p className="text-xs text-primary-foreground/60 mt-0.5">{t(language, 'login.brand.tagline')}</p>
                        </div>
                    </div>
                    <p className="hidden sm:block text-xs text-primary-foreground/50">
                        {t(language, 'login.brand.cities')}
                    </p>
                </div>
                {/* Amber accent line — inspired by Indian government document borders */}
                <div className="h-0.5 bg-amber-500/80" />
            </header>

            {/* ════════════════════════════════════════════════════
                MAIN CONTENT — Photo + Form side by side (desktop)
                                Photo banner + Form stacked (mobile)
                ════════════════════════════════════════════════════ */}
            {/* Content: relative wrapper. Photo is absolute on desktop, static on mobile */}
            <div className="flex-1 relative">

                {/* ── Photo panel ── */}
                {/* Mobile: static 112px banner. Desktop: absolute left strip, full height */}
                <div className="relative md:absolute md:inset-y-0 md:left-0 md:w-80 h-28 md:h-auto overflow-hidden">
                    <img
                        src="/images/kolkata-flood.jpg"
                        alt="Monsoon flooding on a Kolkata street"
                        className="w-full h-full object-cover"
                    />
                    {/* Warm overlay */}
                    <div className="absolute inset-0 bg-gradient-to-b md:bg-gradient-to-r from-amber-900/20 via-transparent to-black/20" />
                    {/* Desktop-only photo caption */}
                    <div className="hidden md:flex absolute bottom-0 left-0 right-0 px-5 py-3 bg-black/50">
                        <p className="text-white/80 text-xs">
                            Kolkata — monsoon flooding disrupts daily commute
                        </p>
                    </div>
                </div>

                {/* ── Form panel ── */}
                {/* Desktop: ml-96 pushes form past the absolute photo */}
                <div className="md:ml-80 min-h-full flex items-start md:items-center justify-center px-6 sm:px-10 md:px-12 py-8 md:py-12 bg-card">
                    <div className="w-full max-w-sm">

                        {authMethod === 'email' ? (
                            <>
                                {/* Heading */}
                                <h2 className="text-2xl font-semibold text-foreground mb-1">
                                    {isSignUp ? t(language, 'login.heading.create') : t(language, 'login.heading.signin')}
                                </h2>
                                <p className="text-muted-foreground text-sm mb-6">
                                    {isSignUp ? t(language, 'login.subheading.create') : t(language, 'login.subheading.signin')}
                                </p>

                                {/* Lockout Banner */}
                                {lockedUntil && new Date() < lockedUntil && (
                                    <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg text-center">
                                        <p className="text-red-700 font-medium text-sm">Account locked</p>
                                        <p className="text-red-600 text-xs">Try again in {lockoutMinutes} minute(s)</p>
                                    </div>
                                )}

                                {/* Error */}
                                {displayError && (
                                    <div className="mb-4 p-3 bg-destructive/10 border border-destructive/20 rounded-lg flex items-start gap-2">
                                        <AlertCircle className="w-4 h-4 text-destructive shrink-0 mt-0.5" />
                                        <p className="text-sm text-destructive">{displayError}</p>
                                    </div>
                                )}

                                {/* Email form — tight spacing, no visual excess */}
                                <form onSubmit={handleEmailSubmit} className="space-y-3">
                                    <div>
                                        <label htmlFor="email" className="block text-sm font-medium text-foreground mb-1">{t(language, 'login.label.email')}</label>
                                        <input
                                            id="email"
                                            type="email"
                                            value={email}
                                            onChange={(e) => setEmail(e.target.value)}
                                            placeholder={t(language, 'login.placeholder.email')}
                                            className="w-full px-3.5 py-2.5 border border-border rounded-lg text-sm text-foreground placeholder:text-muted-foreground bg-background focus:outline-none focus:border-primary focus:ring-2 focus:ring-primary/10 transition-all"
                                            autoComplete="email"
                                        />
                                    </div>
                                    <div>
                                        <label htmlFor="password" className="block text-sm font-medium text-foreground mb-1">{t(language, 'login.label.password')}</label>
                                        <div className="relative">
                                            <input
                                                id="password"
                                                type={showPassword ? 'text' : 'password'}
                                                value={password}
                                                onChange={(e) => setPassword(e.target.value)}
                                                placeholder={isSignUp ? t(language, 'login.placeholder.password.create') : t(language, 'login.placeholder.password.signin')}
                                                className="w-full px-3.5 py-2.5 pr-10 border border-border rounded-lg text-sm text-foreground placeholder:text-muted-foreground bg-background focus:outline-none focus:border-primary focus:ring-2 focus:ring-primary/10 transition-all"
                                                autoComplete={isSignUp ? 'new-password' : 'current-password'}
                                            />
                                            <button
                                                type="button"
                                                onClick={() => setShowPassword(!showPassword)}
                                                className="absolute right-2.5 top-1/2 -translate-y-1/2 p-1 text-muted-foreground hover:text-foreground transition-colors"
                                            >
                                                {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                                            </button>
                                        </div>
                                        {isSignUp && <PasswordStrengthIndicator password={password} />}
                                    </div>
                                    <button
                                        type="submit"
                                        disabled={isLoading || !email || password.length < 8 || (!!lockedUntil && new Date() < lockedUntil)}
                                        className="w-full py-2.5 mt-1 bg-primary hover:bg-primary/90 disabled:opacity-40 disabled:cursor-not-allowed text-primary-foreground font-medium rounded-lg flex items-center justify-center gap-2 transition-all text-sm"
                                    >
                                        {isLoading ? (
                                            <><Loader2 className="w-4 h-4 animate-spin" />{isSignUp ? t(language, 'login.button.creating') : t(language, 'login.button.signingIn')}</>
                                        ) : (
                                            <>{isSignUp ? t(language, 'login.button.create') : t(language, 'login.button.signin')}<ArrowRight className="w-4 h-4" /></>
                                        )}
                                    </button>
                                    {!isSignUp && (
                                        <div className="text-center mt-2">
                                            <a href="/forgot-password" className="text-sm text-blue-600 hover:underline">
                                                Forgot password?
                                            </a>
                                        </div>
                                    )}
                                </form>

                                {/* Divider */}
                                <div className="relative my-5">
                                    <div className="absolute inset-0 flex items-center"><div className="w-full border-t border-border" /></div>
                                    <div className="relative flex justify-center text-xs">
                                        <span className="bg-card px-3 text-muted-foreground">{t(language, 'login.divider.or')}</span>
                                    </div>
                                </div>

                                {/* Google Sign-In */}
                                <div>
                                    {scriptStatus === 'loading' && <div className="h-10 bg-secondary rounded-lg animate-pulse" />}
                                    {scriptStatus === 'error' && !displayError && (
                                        <div className="text-center p-2.5 bg-amber-50 rounded-lg border border-amber-200">
                                            <p className="text-xs text-amber-700">
                                                Google Sign-In unavailable.{' '}
                                                <button onClick={() => window.location.reload()} className="text-amber-600 underline">Refresh</button>
                                            </p>
                                        </div>
                                    )}
                                    {scriptStatus === 'ready' && (
                                        <div ref={googleButtonRef} className="flex justify-center" />
                                    )}
                                </div>

                                {/* Sign up/in toggle + phone link */}
                                <div className="mt-6 text-center text-sm space-y-2">
                                    <p className="text-muted-foreground">
                                        {isSignUp ? (
                                            <>{t(language, 'login.toggle.toSignin')}{' '}
                                                <button type="button" onClick={() => setIsSignUp(false)} className="text-primary font-medium hover:underline">{t(language, 'login.toggle.signin')}</button>
                                            </>
                                        ) : (
                                            <>{t(language, 'login.toggle.toSignup')}{' '}
                                                <button type="button" onClick={() => setIsSignUp(true)} className="text-primary font-medium hover:underline">{t(language, 'login.toggle.signup')}</button>
                                            </>
                                        )}
                                    </p>
                                    <button
                                        type="button"
                                        onClick={() => setAuthMethod('phone')}
                                        className="inline-flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors"
                                    >
                                        <Phone className="w-3 h-3" />
                                        {t(language, 'login.toggle.toPhone')}
                                    </button>
                                </div>
                            </>
                        ) : (
                            <>
                                {/* Phone auth view */}
                                <h2 className="text-2xl font-semibold text-foreground mb-1">{t(language, 'login.phone.heading')}</h2>
                                <p className="text-muted-foreground text-sm mb-6">{t(language, 'login.phone.subheading')}</p>

                                {displayError && (
                                    <div className="mb-4 p-3 bg-destructive/10 border border-destructive/20 rounded-lg flex items-start gap-2">
                                        <AlertCircle className="w-4 h-4 text-destructive shrink-0 mt-0.5" />
                                        <p className="text-sm text-destructive">{displayError}</p>
                                    </div>
                                )}

                                {!otpStep ? (
                                    <form onSubmit={handlePhoneSubmit} className="space-y-4">
                                        <div className="flex border border-border rounded-lg overflow-hidden focus-within:border-primary focus-within:ring-2 focus-within:ring-primary/10 transition-all bg-background">
                                            <select
                                                value={countryCode}
                                                onChange={(e) => setCountryCode(e.target.value)}
                                                className="px-3 py-2.5 bg-secondary border-r border-border text-foreground text-sm focus:outline-none font-medium"
                                            >
                                                <option value="+91">🇮🇳 +91</option>
                                                <option value="+62">🇮🇩 +62</option>
                                                <option value="+65">🇸🇬 +65</option>
                                            </select>
                                            <input
                                                type="tel"
                                                value={phoneNumber}
                                                onChange={(e) => setPhoneNumber(e.target.value.replace(/[^0-9]/g, ''))}
                                                placeholder={t(language, 'login.phone.placeholder')}
                                                className="flex-1 px-3.5 py-2.5 text-foreground placeholder:text-muted-foreground focus:outline-none text-sm bg-transparent"
                                                maxLength={15}
                                            />
                                        </div>
                                        <button
                                            type="submit"
                                            disabled={isPhoneLoading || phoneNumber.length < 8}
                                            className="w-full py-2.5 bg-primary hover:bg-primary/90 disabled:opacity-40 disabled:cursor-not-allowed text-primary-foreground font-medium rounded-lg flex items-center justify-center gap-2 transition-all text-sm"
                                        >
                                            {isPhoneLoading ? (
                                                <><Loader2 className="w-4 h-4 animate-spin" />Starting...</>
                                            ) : (
                                                <>Continue with WhatsApp<ArrowRight className="w-4 h-4" /></>
                                            )}
                                        </button>
                                    </form>
                                ) : (
                                    <div className="space-y-4">
                                        <button
                                            onClick={() => {
                                                setOtpStep(false);
                                                setLoginCode('');
                                                setSessionId('');
                                                setWaLink(null);
                                                setExpiresAt(null);
                                            }}
                                            className="flex items-center gap-1.5 text-muted-foreground hover:text-primary text-sm transition-colors"
                                        >
                                            <ArrowLeft className="w-4 h-4" />Change number
                                        </button>

                                        <p className="text-sm text-muted-foreground">
                                            Send this code to FloodSafe on WhatsApp:
                                        </p>

                                        {/* Prominent code display */}
                                        <div className="text-center py-4 px-6 bg-primary/5 border-2 border-primary/20 rounded-xl">
                                            <p className="text-3xl font-mono font-bold tracking-[0.3em] text-primary">
                                                LOGIN-{loginCode}
                                            </p>
                                        </div>

                                        {/* Open WhatsApp button */}
                                        {waLink && (
                                            <a
                                                href={waLink}
                                                target="_blank"
                                                rel="noopener noreferrer"
                                                className="w-full py-2.5 bg-green-600 hover:bg-green-700 text-white font-medium rounded-lg flex items-center justify-center gap-2 transition-all text-sm"
                                            >
                                                Open WhatsApp
                                            </a>
                                        )}

                                        {/* Polling status */}
                                        <div className="flex items-center justify-center gap-2 text-sm text-muted-foreground">
                                            <Loader2 className="w-4 h-4 animate-spin" />
                                            Waiting for verification...
                                        </div>

                                        {/* Expiry countdown */}
                                        {expiresAt && (
                                            <p className="text-xs text-muted-foreground text-center">
                                                Code expires in {Math.max(0, Math.ceil((expiresAt.getTime() - Date.now()) / 60000))} minutes
                                            </p>
                                        )}
                                    </div>
                                )}
                                <button
                                    type="button"
                                    onClick={() => setAuthMethod('email')}
                                    className="w-full flex items-center justify-center gap-1.5 mt-6 text-sm text-muted-foreground hover:text-foreground transition-colors"
                                >
                                    <ArrowLeft className="w-3.5 h-3.5" />{t(language, 'login.phone.backToEmail')}
                                </button>
                            </>
                        )}

                        {/* Terms footer */}
                        <p className="text-center text-xs text-muted-foreground mt-6 cursor-pointer hover:underline">
                            {t(language, 'login.terms')}
                        </p>
                    </div>
                </div>
            </div>
        </div>
    );
}

export default LoginScreen;
