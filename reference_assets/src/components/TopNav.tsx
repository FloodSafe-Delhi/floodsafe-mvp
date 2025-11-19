import { Menu, Bell, User } from 'lucide-react';
import { Badge } from './ui/badge';

interface TopNavProps {
  onMenuClick?: () => void;
  onNotificationClick?: () => void;
  onProfileClick?: () => void;
  notificationCount?: number;
}

export function TopNav({ 
  onMenuClick, 
  onNotificationClick, 
  onProfileClick,
  notificationCount = 2 
}: TopNavProps) {
  const currentDateTime = new Date().toLocaleString('en-IN', {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit'
  });

  return (
    <header className="fixed top-0 left-0 right-0 bg-blue-600 text-white shadow-md z-50 safe-area-top">
      <div className="flex items-center justify-between px-4 h-14">
        <button
          onClick={onMenuClick}
          className="p-2 -ml-2 min-w-[44px] min-h-[44px] flex items-center justify-center"
          aria-label="Open menu"
        >
          <Menu className="w-6 h-6" />
        </button>
        
        <div className="flex-1 text-center">
          <h1 className="text-lg">FloodSafe Delhi</h1>
          <p className="text-xs opacity-90">{currentDateTime}</p>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={onNotificationClick}
            className="relative p-2 min-w-[44px] min-h-[44px] flex items-center justify-center"
            aria-label={`Notifications, ${notificationCount} unread`}
          >
            <Bell className="w-6 h-6" />
            {notificationCount > 0 && (
              <Badge className="absolute top-1 right-1 bg-red-500 text-white min-w-[18px] h-[18px] flex items-center justify-center p-0 text-xs">
                {notificationCount}
              </Badge>
            )}
          </button>
          
          <button
            onClick={onProfileClick}
            className="p-2 -mr-2 min-w-[44px] min-h-[44px] flex items-center justify-center"
            aria-label="Profile"
          >
            <User className="w-6 h-6" />
          </button>
        </div>
      </div>
    </header>
  );
}
