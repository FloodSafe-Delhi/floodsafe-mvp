/**
 * Verification Reminder Banner
 *
 * Non-blocking banner shown to unverified email users.
 * Displays at the top of the HomeScreen, is dismissible.
 *
 * Features:
 * - Only shows for email/password users who haven't verified
 * - Dismissible (stored in session for current visit)
 * - Resend email button with 60-second cooldown
 * - Rate limit handling (shows message if too many resends)
 */

import { useState, useEffect } from 'react';
import { Mail, X, Loader2, CheckCircle } from 'lucide-react';
import { Button } from './ui/button';
import { useAuth } from '../contexts/AuthContext';
import { useResendVerificationEmail } from '../lib/api/hooks';

interface VerificationReminderBannerProps {
    className?: string;
}

export function VerificationReminderBanner({ className = '' }: VerificationReminderBannerProps) {
    const { user } = useAuth();
    const resendMutation = useResendVerificationEmail();

    const [dismissed, setDismissed] = useState(false);
    const [cooldownSeconds, setCooldownSeconds] = useState(0);
    const [showSuccess, setShowSuccess] = useState(false);

    // Check session storage for dismissal state
    useEffect(() => {
        const wasDismissed = sessionStorage.getItem('verification-banner-dismissed');
        if (wasDismissed === 'true') {
            setDismissed(true);
        }
    }, []);

    // Cooldown timer
    useEffect(() => {
        if (cooldownSeconds <= 0) return;

        const timer = setInterval(() => {
            setCooldownSeconds((prev) => Math.max(0, prev - 1));
        }, 1000);

        return () => clearInterval(timer);
    }, [cooldownSeconds]);

    // Don't show for non-email auth providers or if already verified or dismissed
    if (!user) return null;
    if (user.auth_provider !== 'local') return null;
    if (user.email_verified) return null;
    if (dismissed) return null;

    const handleDismiss = () => {
        setDismissed(true);
        sessionStorage.setItem('verification-banner-dismissed', 'true');
    };

    const handleResend = async () => {
        if (cooldownSeconds > 0 || resendMutation.isPending) return;

        try {
            await resendMutation.mutateAsync();
            setShowSuccess(true);
            setCooldownSeconds(60);

            // Hide success message after 3 seconds
            setTimeout(() => setShowSuccess(false), 3000);
        } catch (error) {
            // Error is handled by mutation state
            console.error('Failed to resend verification email:', error);
        }
    };

    return (
        <div className={`bg-amber-50 border-b border-amber-200 ${className}`}>
            <div className="max-w-7xl mx-auto px-4 py-3">
                <div className="flex items-center justify-between gap-4">
                    <div className="flex items-center gap-3 flex-1 min-w-0">
                        <Mail className="w-5 h-5 text-amber-600 flex-shrink-0" />
                        <div className="min-w-0">
                            <p className="text-sm font-medium text-amber-800">
                                Please verify your email
                            </p>
                            <p className="text-xs text-amber-600 truncate">
                                Check your inbox for the verification link to receive important flood alerts
                            </p>
                        </div>
                    </div>

                    <div className="flex items-center gap-2 flex-shrink-0">
                        {showSuccess ? (
                            <span className="text-sm text-green-600 flex items-center gap-1">
                                <CheckCircle className="w-4 h-4" />
                                Sent!
                            </span>
                        ) : resendMutation.isError ? (
                            <span className="text-xs text-red-600">
                                {(resendMutation.error as Error)?.message?.includes('429')
                                    ? 'Too many requests'
                                    : 'Failed to send'}
                            </span>
                        ) : (
                            <Button
                                variant="ghost"
                                size="sm"
                                onClick={handleResend}
                                disabled={cooldownSeconds > 0 || resendMutation.isPending}
                                className="text-amber-700 hover:text-amber-800 hover:bg-amber-100 text-sm"
                            >
                                {resendMutation.isPending ? (
                                    <Loader2 className="w-4 h-4 animate-spin" />
                                ) : cooldownSeconds > 0 ? (
                                    `Resend (${cooldownSeconds}s)`
                                ) : (
                                    'Resend Email'
                                )}
                            </Button>
                        )}

                        <Button
                            variant="ghost"
                            size="sm"
                            onClick={handleDismiss}
                            className="text-amber-600 hover:text-amber-800 hover:bg-amber-100 p-1"
                            aria-label="Dismiss"
                        >
                            <X className="w-4 h-4" />
                        </Button>
                    </div>
                </div>
            </div>
        </div>
    );
}
