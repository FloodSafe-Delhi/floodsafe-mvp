import { ArrowLeft, Award, MapPin, Plus, LogOut } from 'lucide-react';
import { Card } from '../ui/card';
import { Badge } from '../ui/badge';
import { Button } from '../ui/button';
import { Switch } from '../ui/switch';
import { Label } from '../ui/label';
import { RadioGroup, RadioGroupItem } from '../ui/radio-group';
import { Checkbox } from '../ui/checkbox';
import { Avatar, AvatarFallback } from '../ui/avatar';
import { Progress } from '../ui/progress';
import { Separator } from '../ui/separator';
import { mockUserProfile, mockWatchAreas } from '../../lib/mockData';
import { getAlertIcon } from '../../lib/utils';
import { useState } from 'react';

interface ProfileScreenProps {
  onBack?: () => void;
}

export function ProfileScreen({ onBack }: ProfileScreenProps) {
  const [language, setLanguage] = useState('english');
  const [pushNotifications, setPushNotifications] = useState(true);
  const [smsAlerts, setSmsAlerts] = useState(true);
  const [whatsappUpdates, setWhatsappUpdates] = useState(false);

  const progressToGold = (mockUserProfile.score / 100) * 100;

  return (
    <div className="pb-16 min-h-screen bg-gray-50">
      {/* Profile Header */}
      <div className="bg-gradient-to-br from-blue-600 to-blue-700 text-white p-6">
        <div className="flex items-center gap-4 mb-4">
          <Avatar className="w-16 h-16 bg-white text-blue-600">
            <AvatarFallback className="text-2xl">C</AvatarFallback>
          </Avatar>
          <div className="flex-1">
            <h2 className="text-xl mb-1">{mockUserProfile.role}</h2>
            <p className="text-sm opacity-90">{mockUserProfile.phone}</p>
            <p className="text-xs opacity-75 mt-1">Joined {mockUserProfile.joinDate}</p>
          </div>
        </div>
      </div>

      {/* Reputation Section */}
      <div className="p-4 space-y-4">
        <Card className="p-4">
          <div className="flex items-center justify-between mb-3">
            <h3>Reputation Score</h3>
            <div className="flex items-center gap-2">
              <span className="text-2xl">{mockUserProfile.badge.split(' ')[0]}</span>
              <span className="text-sm">{mockUserProfile.badge.split(' ').slice(1).join(' ')}</span>
            </div>
          </div>

          <div className="flex items-center gap-3 mb-4">
            <span className="text-4xl">{mockUserProfile.score}</span>
            <span className="text-gray-600">/100</span>
          </div>

          <div className="space-y-2 mb-4">
            <div className="flex justify-between text-sm">
              <span>Progress to Gold Badge</span>
              <span className="text-gray-600">22 more reports needed</span>
            </div>
            <Progress value={progressToGold} className="h-2" />
          </div>

          <div className="grid grid-cols-3 gap-3 text-center">
            <div className="p-2 bg-gray-50 rounded">
              <div className="text-xl">{mockUserProfile.reportsSubmitted}</div>
              <div className="text-xs text-gray-600">Submitted</div>
            </div>
            <div className="p-2 bg-green-50 rounded">
              <div className="text-xl text-green-600">{mockUserProfile.reportsVerified}</div>
              <div className="text-xs text-gray-600">Verified</div>
            </div>
            <div className="p-2 bg-yellow-50 rounded">
              <div className="text-xl text-yellow-600">{mockUserProfile.reportsPending}</div>
              <div className="text-xs text-gray-600">Pending</div>
            </div>
          </div>

          <div className="mt-4 p-3 bg-blue-50 border border-blue-200 rounded-lg">
            <p className="text-sm text-blue-800">
              🎁 Monthly Rewards: ₹500 mobile recharge eligible
            </p>
          </div>
        </Card>

        {/* Watch Areas */}
        <Card className="p-4">
          <div className="flex items-center justify-between mb-3">
            <h3>Watch Areas ({mockWatchAreas.length})</h3>
            <Button size="sm" variant="outline">
              <Plus className="w-4 h-4 mr-1" />
              Add
            </Button>
          </div>

          <div className="space-y-2">
            {mockWatchAreas.map((area, index) => (
              <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <div className="flex items-center gap-2">
                  <MapPin className="w-4 h-4 text-gray-600" />
                  <span className="text-sm">{area.name}</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-xl">{getAlertIcon(area.status)}</span>
                  <span className="text-xs capitalize">{area.status}</span>
                </div>
              </div>
            ))}
          </div>
        </Card>

        {/* Notification Preferences */}
        <Card className="p-4">
          <h3 className="mb-4">Notification Preferences</h3>

          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <Label htmlFor="push" className="cursor-pointer">Push notifications</Label>
              <Switch 
                id="push" 
                checked={pushNotifications}
                onCheckedChange={setPushNotifications}
              />
            </div>

            <div className="flex items-center justify-between">
              <Label htmlFor="sms" className="cursor-pointer">SMS alerts</Label>
              <Switch 
                id="sms" 
                checked={smsAlerts}
                onCheckedChange={setSmsAlerts}
              />
            </div>

            <div className="flex items-center justify-between">
              <Label htmlFor="whatsapp" className="cursor-pointer">WhatsApp updates</Label>
              <Switch 
                id="whatsapp" 
                checked={whatsappUpdates}
                onCheckedChange={setWhatsappUpdates}
              />
            </div>

            <Separator />

            <div>
              <Label className="mb-3 block">Alert Types</Label>
              <div className="space-y-2">
                {[
                  { id: 'watch', label: 'Yellow Watch alerts', icon: '🟡' },
                  { id: 'advisory', label: 'Orange Advisory', icon: '🟠' },
                  { id: 'warning', label: 'Red Warning', icon: '🔴' },
                  { id: 'emergency', label: 'Emergency alerts', icon: '⚫' }
                ].map((alert) => (
                  <div key={alert.id} className="flex items-center gap-2">
                    <Checkbox id={alert.id} defaultChecked />
                    <Label htmlFor={alert.id} className="flex items-center gap-2 cursor-pointer">
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
          <h3 className="mb-4">Language</h3>

          <RadioGroup value={language} onValueChange={setLanguage}>
            <div className="flex items-center space-x-2">
              <RadioGroupItem value="english" id="english" />
              <Label htmlFor="english" className="cursor-pointer">English</Label>
            </div>
            <div className="flex items-center space-x-2">
              <RadioGroupItem value="hindi" id="hindi" />
              <Label htmlFor="hindi" className="cursor-pointer">हिन्दी (Hindi)</Label>
            </div>
          </RadioGroup>
        </Card>

        {/* About Section */}
        <Card className="p-4">
          <h3 className="mb-3">About</h3>
          
          <div className="space-y-2 text-sm">
            <p className="text-gray-600">App version: 1.2.0</p>
            <Separator />
            <Button variant="ghost" className="w-full justify-start p-0 h-auto text-blue-600">
              About FloodSafe Delhi
            </Button>
            <Button variant="ghost" className="w-full justify-start p-0 h-auto text-blue-600">
              Privacy Policy
            </Button>
            <Button variant="ghost" className="w-full justify-start p-0 h-auto text-blue-600">
              Terms of Service
            </Button>
            <Button variant="ghost" className="w-full justify-start p-0 h-auto text-blue-600">
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
