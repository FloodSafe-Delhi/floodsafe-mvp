import { FloodAlert, CommunityReport, SafeRoute, UserProfile, WatchArea } from '../types';

export const mockAlerts: FloodAlert[] = [
  {
    id: '1',
    location: 'Connaught Place',
    level: 'advisory',
    color: 'orange',
    title: 'Connaught Place Flood Advisory',
    expectedTime: '6:00 AM - 2:00 PM, Oct 12',
    confidence: 82,
    waterDepth: 'knee',
    coordinates: [28.6315, 77.2167],
    description: 'Flooding likely in 8-12 hours',
    impact: 'Traffic delays, impassable for sedans',
    source: 'Prophet AI Model + 2 community reports + IoT sensor #14',
    isActive: true,
    timeUntil: '8 hours'
  },
  {
    id: '2',
    location: 'Rajendra Nagar',
    level: 'warning',
    color: 'red',
    title: 'Rajendra Nagar - Flooding Imminent',
    expectedTime: 'Now - 8:00 PM, Oct 11',
    confidence: 91,
    waterDepth: 'waist',
    coordinates: [28.6411, 77.1825],
    description: 'Flooding imminent',
    impact: 'Road closures, evacuations recommended',
    source: 'Prophet AI Model + IoT sensor #7 + 5 community reports',
    isActive: true,
    timeUntil: '2 hours'
  },
  {
    id: '3',
    location: 'Karol Bagh',
    level: 'watch',
    color: 'yellow',
    title: 'Karol Bagh - Possible Flooding',
    expectedTime: '12:00 PM - 6:00 PM, Oct 12',
    confidence: 68,
    waterDepth: 'ankle',
    coordinates: [28.6519, 77.1910],
    description: 'Possible flooding',
    impact: 'Minor traffic delays possible',
    source: 'IMD heavy rain forecast + Prophet AI Model',
    isActive: true,
    timeUntil: '18 hours'
  }
];

export const mockCommunityReports: CommunityReport[] = [
  {
    id: '1',
    userId: 'user1',
    userName: 'Rajesh K.',
    timestamp: '2 hours ago',
    location: 'Connaught Place',
    coordinates: [28.6315, 77.2167],
    waterDepth: 'ankle',
    vehiclePassability: 'all',
    verified: true,
    accuracy: 95
  },
  {
    id: '2',
    userId: 'user2',
    userName: 'Priya S.',
    timestamp: '4 hours ago',
    location: 'Connaught Place',
    coordinates: [28.6315, 77.2167],
    waterDepth: 'knee',
    vehiclePassability: 'high-clearance',
    verified: true,
    accuracy: 88
  }
];

export const mockSafeRoutes: SafeRoute[] = [
  {
    id: '1',
    name: 'Via Ring Road',
    additionalTime: '+15 min',
    status: 'Fully clear',
    description: 'Best option for heavy vehicles'
  },
  {
    id: '2',
    name: 'Via Bahadur Shah Zafar Marg',
    additionalTime: '+8 min',
    status: 'Light traffic',
    description: 'Fastest alternative route'
  },
  {
    id: '3',
    name: 'Via Kasturba Gandhi Marg',
    additionalTime: '+12 min',
    status: 'Best option',
    description: 'Recommended for all vehicles'
  }
];

export const mockUserProfile: UserProfile = {
  phone: '+91-XXXX-XXX-456',
  role: 'Commuter',
  joinDate: 'Aug 2025',
  score: 78,
  badge: '🥈 Silver Reporter',
  reportsSubmitted: 15,
  reportsVerified: 12,
  reportsPending: 3
};

export const mockWatchAreas: WatchArea[] = [
  { name: 'Connaught Place', status: 'advisory', color: 'orange' },
  { name: 'Rajendra Nagar', status: 'warning', color: 'red' },
  { name: 'Home (Dwarka)', status: 'safe', color: 'green' }
];

export const mockHotspots = [
  { id: 1, name: 'Connaught Place', coordinates: [28.6315, 77.2167], level: 'advisory' },
  { id: 2, name: 'Rajendra Nagar', coordinates: [28.6411, 77.1825], level: 'warning' },
  { id: 3, name: 'Karol Bagh', coordinates: [28.6519, 77.1910], level: 'watch' },
  { id: 4, name: 'Dwarka', coordinates: [28.5921, 77.0460], level: 'safe' },
  { id: 5, name: 'Nehru Place', coordinates: [28.5494, 77.2501], level: 'safe' },
  { id: 6, name: 'Kashmere Gate', coordinates: [28.6692, 77.2289], level: 'watch' }
];
