import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { User } from '../types';

interface UserContextType {
    user: User | null;
    setUser: (user: User | null) => void;
    isLoading: boolean;
    userId: string | null;
}

const UserContext = createContext<UserContextType | undefined>(undefined);

const DEMO_USER_ID = 'admin'; // For MVP - in production this comes from auth
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export function UserProvider({ children }: { children: ReactNode }) {
    const [user, setUser] = useState<User | null>(null);
    const [isLoading, setIsLoading] = useState(true);

    useEffect(() => {
        // For MVP: Auto-fetch demo user
        // In production: This would check auth tokens and fetch current user
        const fetchDemoUser = async () => {
            try {
                setIsLoading(true);
                const response = await fetch(`${API_URL}/api/leaderboards/top?limit=50`);
                if (!response.ok) throw new Error('Failed to fetch users');

                const users = await response.json();
                const adminUser = users.find((u: User) => u.username === 'admin');

                if (adminUser) {
                    setUser(adminUser);
                } else if (users.length > 0) {
                    // Fallback to first user
                    setUser(users[0]);
                } else {
                    console.warn('No users found in database');
                }
            } catch (error) {
                console.error('Failed to fetch user:', error);
            } finally {
                setIsLoading(false);
            }
        };

        fetchDemoUser();
    }, []);

    const value: UserContextType = {
        user,
        setUser,
        isLoading,
        userId: user?.id || null,
    };

    return <UserContext.Provider value={value}>{children}</UserContext.Provider>;
}

export function useUser() {
    const context = useContext(UserContext);
    if (context === undefined) {
        throw new Error('useUser must be used within a UserProvider');
    }
    return context;
}

/**
 * Hook to get current user ID with a fallback
 * Returns demo user ID if no user is logged in (for MVP)
 */
export function useUserId(): string {
    const { userId } = useUser();
    return userId || DEMO_USER_ID;
}
