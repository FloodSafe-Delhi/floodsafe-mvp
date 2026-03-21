/**
 * Forgot Password Screen for FloodSafe.
 *
 * Allows users to request a password reset email. Shows a generic
 * success message regardless of whether the email exists (no info leakage).
 */

import { useState } from 'react';
import { Shield, ArrowLeft, Loader2, Mail } from 'lucide-react';
import { API_BASE_URL } from '../../lib/api/config';

export function ForgotPasswordScreen() {
    const [email, setEmail] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [submitted, setSubmitted] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!email || !email.includes('@')) {
            setError('Please enter a valid email address.');
            return;
        }
        setError(null);
        setIsLoading(true);
        try {
            await fetch(`${API_BASE_URL}/auth/forgot-password`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ email }),
            });
            // Always show success message — never reveal whether email exists
            setSubmitted(true);
        } catch {
            // Even on network error, show the generic message to prevent info leakage
            setSubmitted(true);
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
                    <a
                        href="/login"
                        className="inline-flex items-center gap-1.5 text-sm text-muted-foreground hover:text-foreground transition-colors mb-6"
                    >
                        <ArrowLeft className="w-4 h-4" />
                        Back to login
                    </a>

                    <h2 className="text-2xl font-semibold text-foreground mb-1">Reset your password</h2>
                    <p className="text-muted-foreground text-sm mb-6">
                        Enter your email and we'll send a reset link if an account exists.
                    </p>

                    {submitted ? (
                        <div className="p-4 bg-green-50 border border-green-200 rounded-lg text-center">
                            <Mail className="w-8 h-8 text-green-600 mx-auto mb-2" />
                            <p className="text-green-800 font-medium text-sm">Reset link sent</p>
                            <p className="text-green-700 text-xs mt-1">
                                If an account exists for <strong>{email}</strong>, a reset link has been sent. Check your inbox.
                            </p>
                            <a
                                href="/login"
                                className="inline-block mt-4 text-sm text-blue-600 hover:underline"
                            >
                                Back to login
                            </a>
                        </div>
                    ) : (
                        <form onSubmit={handleSubmit} className="space-y-4">
                            {error && (
                                <div className="p-3 bg-destructive/10 border border-destructive/20 rounded-lg">
                                    <p className="text-sm text-destructive">{error}</p>
                                </div>
                            )}

                            <div>
                                <label htmlFor="email" className="block text-sm font-medium text-foreground mb-1">
                                    Email address
                                </label>
                                <input
                                    id="email"
                                    type="email"
                                    value={email}
                                    onChange={(e) => setEmail(e.target.value)}
                                    placeholder="you@example.com"
                                    className="w-full px-3.5 py-2.5 border border-border rounded-lg text-sm text-foreground placeholder:text-muted-foreground bg-background focus:outline-none focus:border-primary focus:ring-2 focus:ring-primary/10 transition-all"
                                    autoComplete="email"
                                    autoFocus
                                />
                            </div>

                            <button
                                type="submit"
                                disabled={isLoading || !email}
                                className="w-full py-2.5 bg-primary hover:bg-primary/90 disabled:opacity-40 disabled:cursor-not-allowed text-primary-foreground font-medium rounded-lg flex items-center justify-center gap-2 transition-all text-sm"
                            >
                                {isLoading ? (
                                    <>
                                        <Loader2 className="w-4 h-4 animate-spin" />
                                        Sending...
                                    </>
                                ) : (
                                    'Send Reset Link'
                                )}
                            </button>
                        </form>
                    )}
                </div>
            </div>
        </div>
    );
}

export default ForgotPasswordScreen;
