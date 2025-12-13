import { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { User } from '../types';
import { useAuth, AuthUser } from './AuthContext';

interface UserContextType {
    user: User | null;
    setUser: (user: User | null) => void;
    isLoading: boolean;
    userId: string | null;
}

const UserContext = createContext<UserContextType | undefined>(undefined);

/**
 * Convert AuthUser to User type for backwards compatibility.
 */
function authUserToUser(authUser: AuthUser): User {
    return {
        id: authUser.id,
        username: authUser.username,
        email: authUser.email || undefined,
        phone: authUser.phone || undefined,
        role: authUser.role,
        points: authUser.points,
        level: authUser.level,
        reputation_score: authUser.reputation_score,
        profile_photo_url: authUser.profile_photo_url || undefined,
    };
}

export function UserProvider({ children }: { children: ReactNode }) {
    const { user: authUser, isLoading: authLoading, isAuthenticated } = useAuth();
    const [user, setUser] = useState<User | null>(null);

    // Sync user state with auth context
    useEffect(() => {
        if (isAuthenticated && authUser) {
            setUser(authUserToUser(authUser));
        } else {
            setUser(null);
        }
    }, [authUser, isAuthenticated]);

    const value: UserContextType = {
        user,
        setUser,
        isLoading: authLoading,
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
 * Hook to get current user ID
 * Returns null if no user is loaded (caller should handle this)
 */
export function useUserId(): string | null {
    const { userId } = useUser();
    return userId;
}

/**
 * Hook to check if user is ready for actions that require authentication
 */
export function useUserReady(): { userId: string | null; isLoading: boolean; isReady: boolean } {
    const { userId, isLoading } = useUser();
    return {
        userId,
        isLoading,
        isReady: !isLoading && userId !== null
    };
}
