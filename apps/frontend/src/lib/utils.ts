import { type ClassValue, clsx } from "clsx"
import { twMerge } from "tailwind-merge"
import { AlertLevel, AlertColor } from '../types';

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function getAlertColorClass(color: AlertColor): string {
  const colorMap: Record<AlertColor, string> = {
    green: 'bg-green-500',
    yellow: 'bg-yellow-500',
    orange: 'bg-orange-500',
    red: 'bg-red-500',
    black: 'bg-black'
  };
  return colorMap[color] || 'bg-gray-500';
}

export function getAlertBorderColor(color: AlertColor): string {
  const colorMap: Record<AlertColor, string> = {
    green: 'border-green-500',
    yellow: 'border-yellow-500',
    orange: 'border-orange-500',
    red: 'border-red-500',
    black: 'border-black'
  };
  return colorMap[color] || 'border-gray-500';
}

export function getAlertTextColor(color: AlertColor): string {
  const colorMap: Record<AlertColor, string> = {
    green: 'text-green-700',
    yellow: 'text-yellow-700',
    orange: 'text-orange-700',
    red: 'text-red-700',
    black: 'text-black'
  };
  return colorMap[color] || 'text-gray-700';
}

export function getAlertIcon(level: AlertLevel): string {
  const iconMap: Record<AlertLevel, string> = {
    safe: '🟢',
    watch: '🟡',
    advisory: '🟠',
    warning: '🔴',
    emergency: '⚫'
  };
  return iconMap[level] || '🟢';
}

export function getAlertLabel(level: AlertLevel): string {
  const labelMap: Record<AlertLevel, string> = {
    safe: 'SAFE',
    watch: 'YELLOW WATCH',
    advisory: 'ORANGE ADVISORY',
    warning: 'RED WARNING',
    emergency: 'EMERGENCY'
  };
  return labelMap[level] || 'UNKNOWN';
}

export function getWaterDepthLabel(depth: string): string {
  const depthMap: Record<string, string> = {
    ankle: 'Ankle-deep (<0.3m)',
    knee: 'Knee-deep (0.3-0.6m)',
    waist: 'Waist-deep (0.6-1.2m)',
    impassable: 'Impassable (>1.2m)'
  };
  return depthMap[depth] || 'Unknown';
}
