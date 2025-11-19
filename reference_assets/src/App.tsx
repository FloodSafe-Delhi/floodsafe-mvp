import { useState } from 'react';
import { TopNav } from './components/TopNav';
import { BottomNav } from './components/BottomNav';
import { HomeScreen } from './components/screens/HomeScreen';
import { AlertDetailScreen } from './components/screens/AlertDetailScreen';
import { ReportScreen } from './components/screens/ReportScreen';
import { AlertsListScreen } from './components/screens/AlertsListScreen';
import { ProfileScreen } from './components/screens/ProfileScreen';
import { OfflineIndicator } from './components/OfflineIndicator';
import { FloodAlert } from './types';
import { Toaster } from './components/ui/sonner';
import { toast } from 'sonner@2.0.3';

type Screen = 'home' | 'map' | 'report' | 'alerts' | 'profile' | 'alert-detail';

export default function App() {
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
    toast.success('Report submitted successfully! It will be verified within 30 minutes.');
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
        return <HomeScreen onAlertClick={handleAlertClick} />;
      case 'report':
        return <ReportScreen onBack={handleBackFromReport} onSubmit={handleReportSubmit} />;
      case 'alerts':
        return <AlertsListScreen onAlertClick={handleAlertClick} />;
      case 'profile':
        return <ProfileScreen />;
      default:
        return <HomeScreen onAlertClick={handleAlertClick} />;
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <TopNav 
        onNotificationClick={handleNotificationClick}
        onProfileClick={handleProfileClick}
        notificationCount={2}
      />
      
      <main className="pt-14">
        {renderScreen()}
      </main>

      <BottomNav activeTab={activeTab} onTabChange={(tab) => setActiveTab(tab as Screen)} />
      
      <OfflineIndicator 
        isOffline={isOffline} 
        lastUpdate="3:45 PM"
        onRetry={() => setIsOffline(false)}
      />

      <Toaster position="top-center" />
    </div>
  );
}
