import { AlertLevel, AlertColor } from '../types';
import { getAlertColorClass, getAlertIcon, getAlertLabel } from '../lib/utils';
import { cn } from '../lib/utils';

interface AlertStatusBadgeProps {
  level: AlertLevel;
  color: AlertColor;
  size?: 'small' | 'large';
  showIcon?: boolean;
}

export function AlertStatusBadge({ level, color, size = 'small', showIcon = true }: AlertStatusBadgeProps) {
  const isLarge = size === 'large';
  
  return (
    <div className={cn(
      "rounded-full flex items-center justify-center",
      getAlertColorClass(color),
      "text-white",
      isLarge ? "w-24 h-24 flex-col gap-1" : "w-12 h-12"
    )}>
      {showIcon && (
        <span className={isLarge ? "text-3xl" : "text-2xl"}>
          {getAlertIcon(level)}
        </span>
      )}
      {isLarge && (
        <span className="text-xs uppercase px-2 text-center leading-tight">
          {level === 'safe' ? 'Clear' : level}
        </span>
      )}
    </div>
  );
}
