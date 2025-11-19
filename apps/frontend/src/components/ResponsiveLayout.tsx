import { ReactNode } from 'react';
import { Sidebar } from './Sidebar';
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
        <div className="min-h-screen bg-gray-50 flex">
            {/* Desktop Sidebar */}
            <Sidebar activeTab={activeTab} onTabChange={onTabChange} />

            {/* Main Content Area */}
            <div className="flex-1 flex flex-col md:ml-64 min-h-screen">
                {/* Mobile Top Nav */}
                <div className="md:hidden">
                    <TopNav
                        onNotificationClick={onNotificationClick}
                        onProfileClick={onProfileClick}
                        notificationCount={2}
                    />
                </div>

                {/* Desktop Header (Optional, maybe breadcrumbs later) */}
                <header className="hidden md:flex h-16 bg-white border-b items-center justify-between px-8 sticky top-0 z-40">
                    <h1 className="text-xl font-semibold text-gray-800">
                        {activeTab.charAt(0).toUpperCase() + activeTab.slice(1)}
                    </h1>
                    <div className="flex items-center gap-4">
                        <span className="text-sm text-gray-500">Welcome, Anirudh</span>
                        <div className="w-8 h-8 bg-gray-200 rounded-full"></div>
                    </div>
                </header>

                {/* Content */}
                <main className="flex-1 pt-14 md:pt-0 relative">
                    {children}
                </main>

                {/* Mobile Bottom Nav */}
                <div className="md:hidden">
                    <BottomNav activeTab={activeTab} onTabChange={onTabChange} />
                </div>
            </div>
        </div>
    );
}
