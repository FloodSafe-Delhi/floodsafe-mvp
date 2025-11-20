import { ReactNode } from 'react';
import { BottomNav } from './BottomNav';
import { TopNav } from './TopNav';

interface ResponsiveLayoutProps {
    children: ReactNode;
    activeTab: string;
    onTabChange: (tab: string) => void;
    onNotificationClick: () => void;
    onProfileClick: () => void;
}

export function ResponsiveLayout({
    children,
    activeTab,
    onTabChange,
    onNotificationClick,
    onProfileClick
}: ResponsiveLayoutProps) {
    return (
        <div className="min-h-screen bg-gray-50 flex flex-col">
            {/* Mobile Top Nav */}
            <div className="md:hidden">
                <TopNav
                    onNotificationClick={onNotificationClick}
                    onProfileClick={onProfileClick}
                    notificationCount={2}
                />
            </div>

            {/* Content - Full width, no sidebar */}
            <main className="flex-1 pt-14 md:pt-0 relative">
                {children}
            </main>

            {/* Bottom Nav - Always visible */}
            <BottomNav activeTab={activeTab} onTabChange={onTabChange} />
        </div>
    );
}
