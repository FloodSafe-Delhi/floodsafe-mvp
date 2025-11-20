import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { Card } from '../ui/card';
import { Badge } from '../ui/badge';
import { Button } from '../ui/button';
import { Switch } from '../ui/switch';
import { Label } from '../ui/label';
import { Separator } from '../ui/separator';
import { Avatar, AvatarFallback } from '../ui/avatar';
import { Progress } from '../ui/progress';
import { MapPin, Award, Bell, Globe, Settings, LogOut, Edit, Trash2 } from 'lucide-react';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '../ui/dialog';
import { Input } from '../ui/input';
import { RadioGroup, RadioGroupItem } from '../ui/radio-group';
import { Checkbox } from '../ui/checkbox';
import { toast } from 'sonner';
import { User } from '../../types';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// For demo purposes, using a hardcoded user ID
// In production, this would come from auth context
const DEMO_USER_ID = 'admin'; // Will be created by seed script

interface WatchArea {
  id: string;
  name: string;
  latitude: number;
  longitude: number;
  radius: number;
  created_at: string;
}

// Helper function to normalize user data with defaults
const normalizeUserData = (userData: User): User => {
  return {
    ...userData,
    language: userData.language || 'english',
    notification_push: userData.notification_push ?? true,
    notification_sms: userData.notification_sms ?? false,
    notification_whatsapp: userData.notification_whatsapp ?? false,
    notification_email: userData.notification_email ?? true,
    alert_preferences: userData.alert_preferences || {
      watch: true,
      advisory: true,
      warning: true,
      emergency: true,
    },
  };
};

export function ProfileScreen() {
  const queryClient = useQueryClient();
  const [editDialogOpen, setEditDialogOpen] = useState(false);

  // Fetch user profile
  const { data: rawUser, isLoading } = useQuery<User>({
    queryKey: ['user', DEMO_USER_ID],
    queryFn: async () => {
      // Try to get admin user first
      const usersResponse = await fetch(`${API_URL}/api/leaderboards/top?limit=50`);
      const users = await usersResponse.json();
      const adminUser = users.find((u: User) => u.username === 'admin');

      if (adminUser) {
        return normalizeUserData(adminUser);
      }

      // If no admin user, get first user or show error
      if (users.length > 0) {
        return normalizeUserData(users[0]);
      }

      throw new Error('No users found. Please seed the database.');
    },
  });

  const user = rawUser;

  // Fetch watch areas
  const { data: watchAreas = [] } = useQuery<WatchArea[]>({
    queryKey: ['watchAreas', user?.id],
    queryFn: async () => {
      if (!user?.id) return [];
      const response = await fetch(`${API_URL}/api/watch-areas/user/${user.id}`);
      if (!response.ok) return [];
      return response.json();
    },
    enabled: !!user?.id,
  });

  // Update user mutation
  const updateUserMutation = useMutation({
    mutationFn: async (updates: Partial<User>) => {
      const response = await fetch(`${API_URL}/api/users/${user?.id}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(updates),
      });
      if (!response.ok) throw new Error('Failed to update profile');
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['user'] });
      toast.success('Profile updated successfully!');
    },
    onError: () => {
      toast.error('Failed to update profile');
    },
  });

  const handleNotificationToggle = (field: string, value: boolean) => {
    if (!user) return;
    updateUserMutation.mutate({ [field]: value } as Partial<User>);
  };

  const handleAlertPreferenceToggle = (alertType: string, value: boolean) => {
    if (!user || !user.alert_preferences) return;
    const newPreferences = { ...user.alert_preferences, [alertType]: value };
    updateUserMutation.mutate({
      alert_preferences: newPreferences
    } as Partial<User>);
  };

  const handleLanguageChange = (language: string) => {
    if (!user) return;
    updateUserMutation.mutate({ language } as Partial<User>);
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-lg">Loading profile...</div>
      </div>
    );
  }

  if (!user) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <h2 className="text-xl font-semibold mb-2">No User Found</h2>
          <p className="text-gray-600">Please run the database seed script</p>
        </div>
      </div>
    );
  }

  const getInitials = (name: string | undefined) => {
    if (!name) return '??';
    return name.split(' ').map(n => n[0]).join('').toUpperCase().slice(0, 2);
  };

  // Safe calculations with proper null checks
  const safePoints = user.points || 0;
  const safeLevel = user.level || 1;
  const safeReportsCount = user.reports_count || 0;
  const safeVerifiedCount = user.verified_reports_count || 0;

  const progressToNextLevel = ((safePoints % 100) / 100) * 100;
  const pointsNeeded = 100 - (safePoints % 100);
  const memberSince = new Date(user.created_at).toLocaleDateString('en-US', {
    month: 'short',
    year: 'numeric'
  });

  return (
    <div className="pb-16 md:pb-4 min-h-screen bg-gray-50">
      {/* Profile Header */}
      <div className="bg-gradient-to-br from-blue-600 to-blue-700 text-white p-6">
        <div className="flex items-center gap-4 mb-4">
          <Avatar className="w-16 h-16 bg-white text-blue-600">
            <AvatarFallback className="text-2xl font-semibold">
              {getInitials(user.username)}
            </AvatarFallback>
          </Avatar>
          <div className="flex-1">
            <div className="flex items-center gap-2 mb-1">
              <h2 className="text-xl font-semibold">{user.username}</h2>
              <Badge variant="secondary" className="text-xs">
                {user.role}
              </Badge>
            </div>
            <p className="text-sm opacity-90">{user.email}</p>
            {user.phone && <p className="text-xs opacity-75 mt-1">{user.phone}</p>}
            <p className="text-xs opacity-75 mt-1">Joined {memberSince}</p>
          </div>
          <Dialog open={editDialogOpen} onOpenChange={setEditDialogOpen}>
            <DialogTrigger asChild>
              <Button variant="secondary" size="sm">
                <Edit className="w-4 h-4 mr-1" />
                Edit
              </Button>
            </DialogTrigger>
            <EditProfileDialog user={user} onSave={(updates) => {
              updateUserMutation.mutate(updates);
              setEditDialogOpen(false);
            }} />
          </Dialog>
        </div>
      </div>

      <div className="p-4 space-y-4">
        {/* Reputation Section */}
        <Card className="p-4">
          <div className="flex items-center justify-between mb-3">
            <h3 className="font-semibold text-lg flex items-center gap-2">
              <Award className="w-5 h-5 text-yellow-600" />
              Reputation Score
            </h3>
            <div className="text-right">
              <div className="text-2xl font-bold text-blue-600">Level {safeLevel}</div>
              <div className="text-sm text-gray-600">{safePoints} points</div>
            </div>
          </div>

          <div className="space-y-2 mb-4">
            <div className="flex justify-between text-sm">
              <span>Progress to Level {safeLevel + 1}</span>
              <span className="text-gray-600">{pointsNeeded} more points</span>
            </div>
            <Progress value={progressToNextLevel} className="h-2" />
          </div>

          <div className="grid grid-cols-3 gap-3 text-center">
            <div className="p-3 bg-gray-50 rounded-lg">
              <div className="text-2xl font-bold">{safeReportsCount}</div>
              <div className="text-xs text-gray-600">Submitted</div>
            </div>
            <div className="p-3 bg-green-50 rounded-lg">
              <div className="text-2xl font-bold text-green-600">{safeVerifiedCount}</div>
              <div className="text-xs text-gray-600">Verified</div>
            </div>
            <div className="p-3 bg-yellow-50 rounded-lg">
              <div className="text-2xl font-bold text-yellow-600">
                {safeReportsCount - safeVerifiedCount}
              </div>
              <div className="text-xs text-gray-600">Pending</div>
            </div>
          </div>

          {user.badges && user.badges.length > 0 && (
            <div className="mt-4 p-3 bg-blue-50 border border-blue-200 rounded-lg">
              <div className="text-sm font-medium text-blue-900 mb-2">Badges Earned</div>
              <div className="flex flex-wrap gap-2">
                {user.badges.map((badge, idx) => (
                  <Badge key={idx} variant="secondary" className="bg-blue-100 text-blue-800">
                    {badge}
                  </Badge>
                ))}
              </div>
            </div>
          )}
        </Card>

        {/* Watch Areas */}
        <Card className="p-4">
          <div className="flex items-center justify-between mb-3">
            <h3 className="font-semibold text-lg flex items-center gap-2">
              <MapPin className="w-5 h-5 text-gray-600" />
              Watch Areas ({watchAreas.length})
            </h3>
          </div>

          {watchAreas.length > 0 ? (
            <div className="space-y-2">
              {watchAreas.map((area) => (
                <div key={area.id} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                  <div className="flex items-center gap-2 flex-1">
                    <MapPin className="w-4 h-4 text-gray-600" />
                    <div>
                      <div className="text-sm font-medium">{area.name}</div>
                      <div className="text-xs text-gray-500">
                        Radius: {(area.radius / 1000).toFixed(1)}km
                      </div>
                    </div>
                  </div>
                  <Button size="sm" variant="ghost">
                    <Trash2 className="w-4 h-4 text-red-500" />
                  </Button>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-6 text-gray-500">
              <MapPin className="w-12 h-12 mx-auto mb-2 opacity-50" />
              <p className="text-sm">No watch areas yet</p>
              <p className="text-xs mt-1">Add locations to monitor for alerts</p>
            </div>
          )}
        </Card>

        {/* Notification Preferences */}
        <Card className="p-4">
          <h3 className="font-semibold text-lg flex items-center gap-2 mb-4">
            <Bell className="w-5 h-5 text-gray-600" />
            Notification Preferences
          </h3>

          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <Label htmlFor="push" className="cursor-pointer font-normal">
                Push notifications
              </Label>
              <Switch
                id="push"
                checked={user.notification_push ?? true}
                onCheckedChange={(checked) => handleNotificationToggle('notification_push', checked)}
              />
            </div>

            <div className="flex items-center justify-between">
              <Label htmlFor="sms" className="cursor-pointer font-normal">
                SMS alerts
              </Label>
              <Switch
                id="sms"
                checked={user.notification_sms ?? false}
                onCheckedChange={(checked) => handleNotificationToggle('notification_sms', checked)}
              />
            </div>

            <div className="flex items-center justify-between">
              <Label htmlFor="whatsapp" className="cursor-pointer font-normal">
                WhatsApp updates
              </Label>
              <Switch
                id="whatsapp"
                checked={user.notification_whatsapp ?? false}
                onCheckedChange={(checked) => handleNotificationToggle('notification_whatsapp', checked)}
              />
            </div>

            <div className="flex items-center justify-between">
              <Label htmlFor="email" className="cursor-pointer font-normal">
                Email notifications
              </Label>
              <Switch
                id="email"
                checked={user.notification_email ?? true}
                onCheckedChange={(checked) => handleNotificationToggle('notification_email', checked)}
              />
            </div>

            <Separator />

            <div>
              <Label className="mb-3 block font-semibold">Alert Types</Label>
              <div className="space-y-2">
                {[
                  { id: 'watch', label: 'Yellow Watch alerts', icon: '🟡' },
                  { id: 'advisory', label: 'Orange Advisory', icon: '🟠' },
                  { id: 'warning', label: 'Red Warning', icon: '🔴' },
                  { id: 'emergency', label: 'Emergency alerts', icon: '⚫' }
                ].map((alert) => (
                  <div key={alert.id} className="flex items-center gap-2">
                    <Checkbox
                      id={alert.id}
                      checked={user.alert_preferences?.[alert.id as keyof typeof user.alert_preferences] ?? true}
                      onCheckedChange={(checked) =>
                        handleAlertPreferenceToggle(alert.id, checked as boolean)
                      }
                    />
                    <Label htmlFor={alert.id} className="flex items-center gap-2 cursor-pointer font-normal">
                      <span>{alert.icon}</span>
                      <span className="text-sm">{alert.label}</span>
                    </Label>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </Card>

        {/* Language Selection */}
        <Card className="p-4">
          <h3 className="font-semibold text-lg flex items-center gap-2 mb-4">
            <Globe className="w-5 h-5 text-gray-600" />
            Language
          </h3>

          <RadioGroup value={user.language || 'english'} onValueChange={handleLanguageChange}>
            <div className="flex items-center space-x-2">
              <RadioGroupItem value="english" id="english" />
              <Label htmlFor="english" className="cursor-pointer font-normal">English</Label>
            </div>
            <div className="flex items-center space-x-2">
              <RadioGroupItem value="hindi" id="hindi" />
              <Label htmlFor="hindi" className="cursor-pointer font-normal">हिन्दी (Hindi)</Label>
            </div>
          </RadioGroup>
        </Card>

        {/* About Section */}
        <Card className="p-4">
          <h3 className="font-semibold text-lg flex items-center gap-2 mb-3">
            <Settings className="w-5 h-5 text-gray-600" />
            About
          </h3>

          <div className="space-y-2 text-sm">
            <p className="text-gray-600">App version: 1.0.0 (MVP)</p>
            <Separator />
            <Button variant="ghost" className="w-full justify-start p-0 h-auto text-blue-600 font-normal">
              About FloodSafe
            </Button>
            <Button variant="ghost" className="w-full justify-start p-0 h-auto text-blue-600 font-normal">
              Privacy Policy
            </Button>
            <Button variant="ghost" className="w-full justify-start p-0 h-auto text-blue-600 font-normal">
              Terms of Service
            </Button>
            <Button variant="ghost" className="w-full justify-start p-0 h-auto text-blue-600 font-normal">
              Contact Support
            </Button>
          </div>
        </Card>

        {/* Logout */}
        <Button variant="destructive" className="w-full">
          <LogOut className="w-4 h-4 mr-2" />
          Logout
        </Button>
      </div>
    </div>
  );
}

// Edit Profile Dialog Component
function EditProfileDialog({ user, onSave }: { user: User; onSave: (updates: Partial<User>) => void }) {
  const [username, setUsername] = useState(user.username);
  const [email, setEmail] = useState(user.email);
  const [phone, setPhone] = useState(user.phone || '');

  const handleSave = () => {
    const updates: Partial<User> = {};
    if (username !== user.username) updates.username = username;
    if (email !== user.email) updates.email = email;
    if (phone !== user.phone) updates.phone = phone;

    if (Object.keys(updates).length > 0) {
      onSave(updates);
    }
  };

  return (
    <DialogContent>
      <DialogHeader>
        <DialogTitle>Edit Profile</DialogTitle>
      </DialogHeader>
      <div className="space-y-4 py-4">
        <div className="space-y-2">
          <Label htmlFor="edit-username">Username</Label>
          <Input
            id="edit-username"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            placeholder="Enter username"
          />
        </div>
        <div className="space-y-2">
          <Label htmlFor="edit-email">Email</Label>
          <Input
            id="edit-email"
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="Enter email"
          />
        </div>
        <div className="space-y-2">
          <Label htmlFor="edit-phone">Phone Number</Label>
          <Input
            id="edit-phone"
            type="tel"
            value={phone}
            onChange={(e) => setPhone(e.target.value)}
            placeholder="Enter phone number"
          />
        </div>
        <Button onClick={handleSave} className="w-full">
          Save Changes
        </Button>
      </div>
    </DialogContent>
  );
}
