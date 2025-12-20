import React, { createContext, useContext, useState, useCallback, useRef, useEffect } from 'react';

interface VoiceGuidanceContextValue {
    isEnabled: boolean;
    setEnabled: (enabled: boolean) => void;
    speak: (text: string, priority?: 'normal' | 'high') => void;
    stop: () => void;
}

const VoiceGuidanceContext = createContext<VoiceGuidanceContextValue | null>(null);

export function VoiceGuidanceProvider({ children }: { children: React.ReactNode }) {
    const [isEnabled, setIsEnabledState] = useState(true);
    const synthRef = useRef<SpeechSynthesis | null>(null);
    const queueRef = useRef<string[]>([]);
    const isSpeakingRef = useRef(false);

    // Initialize speech synthesis
    useEffect(() => {
        if (typeof window !== 'undefined' && window.speechSynthesis) {
            synthRef.current = window.speechSynthesis;
        }
    }, []);

    const processQueue = useCallback(() => {
        if (!synthRef.current || !isEnabled || isSpeakingRef.current) return;
        if (queueRef.current.length === 0) return;

        const text = queueRef.current.shift()!;
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.rate = 1.0;
        utterance.pitch = 1.0;
        utterance.volume = 1.0;

        // Prefer English voice
        const voices = synthRef.current.getVoices();
        const englishVoice = voices.find(v => v.lang.startsWith('en'));
        if (englishVoice) {
            utterance.voice = englishVoice;
        }

        utterance.onstart = () => { isSpeakingRef.current = true; };
        utterance.onend = () => {
            isSpeakingRef.current = false;
            processQueue();
        };
        utterance.onerror = () => {
            isSpeakingRef.current = false;
            processQueue();
        };

        synthRef.current.speak(utterance);
    }, [isEnabled]);

    const speak = useCallback((text: string, priority: 'normal' | 'high' = 'normal') => {
        if (!isEnabled || !synthRef.current) return;

        if (priority === 'high') {
            // Clear queue and speak immediately
            queueRef.current = [];
            synthRef.current.cancel();
            isSpeakingRef.current = false;
        }

        queueRef.current.push(text);
        processQueue();
    }, [isEnabled, processQueue]);

    const stop = useCallback(() => {
        queueRef.current = [];
        synthRef.current?.cancel();
        isSpeakingRef.current = false;
    }, []);

    const setEnabled = useCallback((enabled: boolean) => {
        setIsEnabledState(enabled);
        localStorage.setItem('floodsafe_voice_enabled', String(enabled));
        if (!enabled) {
            stop();
        }
    }, [stop]);

    // Load preference from localStorage on mount
    useEffect(() => {
        const saved = localStorage.getItem('floodsafe_voice_enabled');
        if (saved !== null) {
            setIsEnabledState(saved === 'true');
        }
    }, []);

    return (
        <VoiceGuidanceContext.Provider value={{ isEnabled, setEnabled, speak, stop }}>
            {children}
        </VoiceGuidanceContext.Provider>
    );
}

export function useVoiceGuidance() {
    const context = useContext(VoiceGuidanceContext);
    if (!context) {
        throw new Error('useVoiceGuidance must be used within VoiceGuidanceProvider');
    }
    return context;
}
