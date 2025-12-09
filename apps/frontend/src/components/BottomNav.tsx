import { Home, Map as MapIcon, PlusCircle, Bell, User } from 'lucide-react';
import { cn } from '../lib/utils';

interface BottomNavProps {
    activeTab: string;
    onTabChange: (tab: string) => void;
}

export function BottomNav({ activeTab, onTabChange }: BottomNavProps) {
    const tabs = [
        { id: 'home', icon: Home, label: 'Home' },
        { id: 'map', icon: MapIcon, label: 'Flood Atlas' },
        { id: 'report', icon: PlusCircle, label: 'Report', primary: true },
        { id: 'alerts', icon: Bell, label: 'Alerts' },
        { id: 'profile', icon: User, label: 'Profile' },
    ];

    return (
        <nav
            data-bottom-nav
            className="fixed bottom-0 left-0 right-0 bg-white border-t pb-safe-area-bottom z-50"
        >
            <div className="flex items-center justify-around h-16">
                {tabs.map((tab) => {
                    const Icon = tab.icon;
                    const isActive = activeTab === tab.id;

                    if (tab.primary) {
                        return (
                            <button
                                key={tab.id}
                                onClick={() => onTabChange(tab.id)}
                                className="flex flex-col items-center justify-center"
                                style={{ marginTop: '-68px' }}
                            >
                                <div className="w-40 h-40 bg-blue-600 rounded-full shadow-lg flex items-center justify-center text-white hover:bg-blue-700 transition-colors">
                                    <Icon className="w-16 h-16" />
                                </div>
                                <span className="text-[11px] mt-2 font-black text-blue-600">{tab.label}</span>
                            </button>
                        );
                    }

                    return (
                        <button
                            key={tab.id}
                            onClick={() => onTabChange(tab.id)}
                            className={cn(
                                "flex flex-col items-center w-full h-full space-y-1",
                                isActive ? "text-blue-600" : "text-gray-500 hover:text-gray-700"
                            )}
                        >
                            <Icon className={cn("w-5 h-5", isActive && "fill-current")} />
                            <span className="text-[10px] font-medium">{tab.label}</span>
                        </button>
                    );
                })}
            </div>
        </nav>
    );
}
