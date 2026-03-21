/**
 * Reset Password Screen for FloodSafe.
 *
 * Accepts a token from the URL query param and allows the user to
 * set a new password. Shows a live password strength checklist.
 */

import { useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import { Shield, Loader2, Eye, EyeOff, AlertCircle } from 'lucide-react';
import { API_BASE_URL } from '../../lib/api/config';

interface PasswordRule {
    label: string;
    met: boolean;
}

function getPasswordRules(password: string): PasswordRule[] {
    return [
        { label: 'At least 8 characters', met: password.length >= 8 },
        { label: 'Uppercase letter', met: /[A-Z]/.test(password) },
        { label: 'Lowercase letter', met: /[a-z]/.test(password) },
        { label: 'Number', met: /\d/.test(password) },
        { label: 'Special character', met: /[^A-Za-z0-9]/.test(password) },
    ];
}

function PasswordStrengthChecklist({ password }: { password: string }) {
    if (!password) return null;
    const rules = getPasswordRules(password);
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

function isPasswordStrong(password: string): boolean {
    return getPasswordRules(password).every((r) => r.met);
}

export function ResetPasswordScreen() {
    const [searchParams] = useSearchParams();
    const token = searchParams.get('token') || '';

    const [password, setPassword] = useState('');
    const [showPassword, setShowPassword] = useState(false);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [success, setSuccess] = useState(false);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setError(null);

        if (!token) {
            setError('Invalid or missing reset token. Please request a new reset link.');
            return;
        }
        if (!isPasswordStrong(password)) {
            setError('Please meet all password requirements before submitting.');
            return;
        }

        setIsLoading(true);
        try {
            const response = await fetch(`${API_BASE_URL}/auth/reset-password`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ token, new_password: password }),
            });

            if (!response.ok) {
                let message = 'Failed to reset password.';
                try {
                    const data = await response.json();
                    if (data.detail) {
                        message = typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail);
                    }
                } catch {
                    // ignore
                }
                throw new Error(message);
            }

            // Remove token from URL so it can't be reused from browser history
            history.replaceState(null, '', '/reset-password');
            setSuccess(true);

            // Redirect to login after a short delay
            setTimeout(() => {
                window.location.href = '/login';
            }, 2500);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'An error occurred. Please try again.');
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="min-h-screen w-full flex flex-col bg-background">
            {/* Brand Bar */}
            <header className="bg-primary text-primary-foreground shrink-0">
                <div className="max-w-5xl mx-auto px-5 py-4 flex items-center gap-3">
                    <div className="w-9 h-9 rounded-lg border border-primary-foreground/20 flex items-center justify-center">
                        <Shield className="w-5 h-5" />
                    </div>
                    <div>
                        <h1 className="text-base font-bold tracking-tight leading-none">FloodSafe</h1>
                        <p className="text-xs text-primary-foreground/60 mt-0.5">Community flood monitoring</p>
                    </div>
                </div>
                <div className="h-0.5 bg-amber-500/80" />
            </header>

            {/* Form */}
            <div className="flex-1 flex items-start md:items-center justify-center px-6 sm:px-10 py-12 bg-card">
                <div className="w-full max-w-sm">
                    <h2 className="text-2xl font-semibold text-foreground mb-1">Set a new password</h2>
                    <p className="text-muted-foreground text-sm mb-6">
                        Choose a strong password for your account.
                    </p>

                    {success ? (
                        <div className="p-4 bg-green-50 border border-green-200 rounded-lg text-center">
                            <p className="text-green-800 font-medium text-sm">Password updated</p>
                            <p className="text-green-700 text-xs mt-1">
                                Your password has been reset. Redirecting you to login...
                            </p>
                        </div>
                    ) : !token ? (
                        <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
                            <div className="flex items-start gap-2">
                                <AlertCircle className="w-4 h-4 text-red-600 shrink-0 mt-0.5" />
                                <div>
                                    <p className="text-red-800 font-medium text-sm">Invalid reset link</p>
                                    <p className="text-red-700 text-xs mt-1">
                                        This link is missing a token.{' '}
                                        <a href="/forgot-password" className="underline">
                                            Request a new one
                                        </a>
                                        .
                                    </p>
                                </div>
                            </div>
                        </div>
                    ) : (
                        <form onSubmit={handleSubmit} className="space-y-4">
                            {error && (
                                <div className="p-3 bg-destructive/10 border border-destructive/20 rounded-lg flex items-start gap-2">
                                    <AlertCircle className="w-4 h-4 text-destructive shrink-0 mt-0.5" />
                                    <p className="text-sm text-destructive">{error}</p>
                                </div>
                            )}

                            <div>
                                <label htmlFor="new-password" className="block text-sm font-medium text-foreground mb-1">
                                    New password
                                </label>
                                <div className="relative">
                                    <input
                                        id="new-password"
                                        type={showPassword ? 'text' : 'password'}
                                        value={password}
                                        onChange={(e) => setPassword(e.target.value)}
                                        placeholder="Enter new password"
                                        className="w-full px-3.5 py-2.5 pr-10 border border-border rounded-lg text-sm text-foreground placeholder:text-muted-foreground bg-background focus:outline-none focus:border-primary focus:ring-2 focus:ring-primary/10 transition-all"
                                        autoComplete="new-password"
                                        autoFocus
                                    />
                                    <button
                                        type="button"
                                        onClick={() => setShowPassword(!showPassword)}
                                        className="absolute right-2.5 top-1/2 -translate-y-1/2 p-1 text-muted-foreground hover:text-foreground transition-colors"
                                    >
                                        {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                                    </button>
                                </div>
                                <PasswordStrengthChecklist password={password} />
                            </div>

                            <button
                                type="submit"
                                disabled={isLoading || !isPasswordStrong(password)}
                                className="w-full py-2.5 mt-1 bg-primary hover:bg-primary/90 disabled:opacity-40 disabled:cursor-not-allowed text-primary-foreground font-medium rounded-lg flex items-center justify-center gap-2 transition-all text-sm"
                            >
                                {isLoading ? (
                                    <>
                                        <Loader2 className="w-4 h-4 animate-spin" />
                                        Resetting...
                                    </>
                                ) : (
                                    'Reset Password'
                                )}
                            </button>

                            <p className="text-center text-sm">
                                <a href="/login" className="text-blue-600 hover:underline">
                                    Back to login
                                </a>
                            </p>
                        </form>
                    )}
                </div>
            </div>
        </div>
    );
}

export default ResetPasswordScreen;
