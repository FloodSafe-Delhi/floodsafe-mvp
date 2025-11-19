import { Badge } from './ui/badge';
import { cn } from '../lib/utils';

interface AlertStatusBadgeProps {
    level: string;
    color: string;
    size?: 'small' | 'large';
}

export function AlertStatusBadge({ level, color, size = 'small' }: AlertStatusBadgeProps) {
    const colorClasses = {
        red: 'bg-red-100 text-red-800 border-red-200',
        orange: 'bg-orange-100 text-orange-800 border-orange-200',
        yellow: 'bg-yellow-100 text-yellow-800 border-yellow-200',
        green: 'bg-green-100 text-green-800 border-green-200',
    };

    return (
        <Badge
            variant="outline"
            className={cn(
                "uppercase tracking-wider font-bold",
                colorClasses[color as keyof typeof colorClasses] || colorClasses.green,
                size === 'large' ? 'px-4 py-1 text-sm' : 'text-xs'
            )}
        >
            {level}
        </Badge>
    );
}
