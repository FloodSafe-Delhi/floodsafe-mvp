import { useState } from 'react';
import { ResponsiveLayout } from './components/ResponsiveLayout';
import { HomeScreen } from './components/screens/HomeScreen';
import { FloodAtlasScreen } from './components/screens/FloodAtlasScreen';
import { ReportScreen } from './components/screens/ReportScreen';
import { AlertDetailScreen, AlertsListScreen, ProfileScreen } from './components/screens/Placeholders';
import { OfflineIndicator } from './components/OfflineIndicator';
import { FloodAlert } from './types';
import { Toaster } from './components/ui/sonner';
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { CityProvider } from './contexts/CityContext';

const queryClient = new QueryClient();

type Screen = 'home' | 'map' | 'report' | 'alerts' | 'profile' | 'alert-detail';

function FloodSafeApp() {
    const [activeTab, setActiveTab] = useState<Screen>('home');
    const [selectedAlert, setSelectedAlert] = useState<FloodAlert | null>(null);
    const [isOffline, setIsOffline] = useState(false);

    const handleAlertClick = (alert: FloodAlert) => {
        setSelectedAlert(alert);
        setActiveTab('alert-detail');
    };

    const handleBackFromDetail = () => {
        setSelectedAlert(null);
        setActiveTab('home');
    };

    const handleBackFromReport = () => {
        setActiveTab('home');
    };

    const handleReportSubmit = () => {
        setActiveTab('home');
    };

    const handleNotificationClick = () => {
        setActiveTab('alerts');
    };

    const handleProfileClick = () => {
        setActiveTab('profile');
    };

    const renderScreen = () => {
        switch (activeTab) {
            case 'home':
                return <HomeScreen onAlertClick={handleAlertClick} />;
            case 'alert-detail':
                return selectedAlert ? (
                    <AlertDetailScreen alert={selectedAlert} onBack={handleBackFromDetail} />
                ) : (
                    <HomeScreen onAlertClick={handleAlertClick} />
                );
            case 'map':
                return <FloodAtlasScreen />;
            case 'report':
                return <ReportScreen onBack={handleBackFromReport} onSubmit={handleReportSubmit} onSelectLocation={() => setActiveTab('map')} />;
            case 'alerts':
                return <AlertsListScreen onAlertClick={handleAlertClick} />;
            case 'profile':
                return <ProfileScreen />;
            default:
                return <HomeScreen onAlertClick={handleAlertClick} />;
        }
    };

    return (
        <ResponsiveLayout
            activeTab={activeTab}
            onTabChange={(tab) => setActiveTab(tab as Screen)}
            onNotificationClick={handleNotificationClick}
            onProfileClick={handleProfileClick}
        >
            {renderScreen()}

            <OfflineIndicator
                isOffline={isOffline}
                lastUpdate="Just now"
                onRetry={() => setIsOffline(false)}
            />

            <Toaster position="top-center" />
        </ResponsiveLayout>
    );
}

export default function App() {
    return (
        <QueryClientProvider client={queryClient}>
            <CityProvider>
                <FloodSafeApp />
            </CityProvider>
        </QueryClientProvider>
    );
}
