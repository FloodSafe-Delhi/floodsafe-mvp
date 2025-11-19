import { Home, Map as MapIcon, PlusCircle, Bell, User, LogOut } from 'lucide-react';
import { cn } from '../lib/utils';
import { Button } from './ui/button';

interface SidebarProps {
    activeTab: string;
    onTabChange: (tab: string) => void;
}

export function Sidebar({ activeTab, onTabChange }: SidebarProps) {
    const tabs = [
        { id: 'home', icon: Home, label: 'Dashboard' },
        { id: 'map', icon: MapIcon, label: 'Flood Atlas' },
        { id: 'alerts', icon: Bell, label: 'Alerts' },
        { id: 'profile', icon: User, label: 'Profile' },
    ];

    return (
        <aside className="hidden md:flex flex-col w-64 h-screen bg-white border-r fixed left-0 top-0 z-50">
            {/* Logo */}
            <div className="h-16 flex items-center px-6 border-b">
                <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center text-white font-bold mr-3">
                    FS
                </div>
                <span className="font-bold text-xl text-blue-900">FloodSafe</span>
            </div>

            {/* Navigation */}
            <nav className="flex-1 p-4 space-y-2">
                {tabs.map((tab) => {
                    const Icon = tab.icon;
                    const isActive = activeTab === tab.id;

                    return (
                        <button
                            key={tab.id}
                            onClick={() => onTabChange(tab.id)}
                            className={cn(
                                "w-full flex items-center gap-3 px-4 py-3 rounded-lg transition-colors",
                                isActive
                                    ? "bg-blue-50 text-blue-600 font-medium"
                                    : "text-gray-600 hover:bg-gray-50 hover:text-gray-900"
                            )}
                        >
                            <Icon className={cn("w-5 h-5", isActive && "fill-current")} />
                            {tab.label}
                        </button>
                    );
                })}

                <div className="pt-4 mt-4 border-t">
                    <button
                        onClick={() => onTabChange('report')}
                        className={cn(
                            "w-full flex items-center gap-3 px-4 py-3 rounded-lg transition-colors bg-blue-600 text-white hover:bg-blue-700 shadow-md"
                        )}
                    >
                        <PlusCircle className="w-5 h-5" />
                        Report Flood
                    </button>
                </div>
            </nav>

            {/* Footer */}
            <div className="p-4 border-t">
                <Button variant="ghost" className="w-full justify-start text-red-500 hover:text-red-600 hover:bg-red-50">
                    <LogOut className="w-4 h-4 mr-2" />
                    Logout
                </Button>
            </div>
        </aside>
    );
}
