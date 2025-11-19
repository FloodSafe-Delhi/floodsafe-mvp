import { Bell, User } from 'lucide-react';
import { Button } from './ui/button';

interface TopNavProps {
    onNotificationClick: () => void;
    onProfileClick: () => void;
    notificationCount?: number;
}

export function TopNav({ onNotificationClick, onProfileClick, notificationCount = 0 }: TopNavProps) {
    return (
        <header className="fixed top-0 left-0 right-0 h-14 bg-white border-b z-50 px-4 flex items-center justify-between">
            <div className="flex items-center gap-2">
                <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center text-white font-bold">
                    FS
                </div>
                <span className="font-bold text-lg text-blue-900">FloodSafe</span>
            </div>

            <div className="flex items-center gap-2">
                <Button variant="ghost" size="icon" onClick={onNotificationClick} className="relative">
                    <Bell className="w-5 h-5 text-gray-600" />
                    {notificationCount > 0 && (
                        <span className="absolute top-2 right-2 w-2 h-2 bg-red-500 rounded-full" />
                    )}
                </Button>
                <Button variant="ghost" size="icon" onClick={onProfileClick}>
                    <User className="w-5 h-5 text-gray-600" />
                </Button>
            </div>
        </header>
    );
}
