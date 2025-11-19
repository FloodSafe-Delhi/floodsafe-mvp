import { ArrowLeft, Share2, Navigation, Clock, Droplet, AlertCircle } from 'lucide-react';
import { Card } from '../ui/card';
import { Badge } from '../ui/badge';
import { Button } from '../ui/button';
import { Progress } from '../ui/progress';
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from '../ui/accordion';
import { FloodAlert } from '../../types';
import { getAlertIcon, getAlertLabel, getAlertColorClass, getWaterDepthLabel } from '../../lib/utils';
import { mockSafeRoutes, mockCommunityReports } from '../../lib/mockData';
import { Avatar, AvatarFallback } from '../ui/avatar';

interface AlertDetailScreenProps {
  alert: FloodAlert;
  onBack: () => void;
}

export function AlertDetailScreen({ alert, onBack }: AlertDetailScreenProps) {
  const reports = mockCommunityReports.filter(r => r.location === alert.location);

  return (
    <div className="pb-16 min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow-sm sticky top-14 z-40">
        <div className="flex items-center justify-between px-4 h-14">
          <button
            onClick={onBack}
            className="p-2 -ml-2 min-w-[44px] min-h-[44px] flex items-center justify-center"
            aria-label="Go back"
          >
            <ArrowLeft className="w-6 h-6" />
          </button>
          
          <h1 className="flex-1 text-center px-4 truncate">
            {alert.title}
          </h1>

          <button
            className="p-2 -mr-2 min-w-[44px] min-h-[44px] flex items-center justify-center"
            aria-label="Share alert"
          >
            <Share2 className="w-6 h-6" />
          </button>
        </div>
      </div>

      {/* Alert Banner */}
      <div className={`${getAlertColorClass(alert.color)} text-white p-6`}>
        <div className="flex items-center gap-4">
          <span className="text-4xl">{getAlertIcon(alert.level)}</span>
          <div className="flex-1">
            <h2 className="text-xl mb-1">{getAlertLabel(alert.level)}</h2>
            <p className="text-sm opacity-90">{alert.description}</p>
          </div>
        </div>
        
        <div className="mt-4">
          <p className="text-xs mb-1 opacity-80">Prophet Prediction Confidence</p>
          <div className="flex items-center gap-2">
            <Progress value={alert.confidence} className="h-2 flex-1 bg-white/30" />
            <span>{alert.confidence}%</span>
          </div>
        </div>
      </div>

      {/* Alert Details */}
      <div className="p-4 space-y-4">
        <Card className="p-4">
          <h3 className="mb-3">Alert Details</h3>
          
          <div className="space-y-3 text-sm">
            <div className="flex items-start gap-3">
              <Clock className="w-5 h-5 text-gray-500 mt-0.5" />
              <div className="flex-1">
                <p className="text-gray-600">Expected Time</p>
                <p>{alert.expectedTime}</p>
              </div>
            </div>

            <div className="flex items-start gap-3">
              <Droplet className="w-5 h-5 text-gray-500 mt-0.5" />
              <div className="flex-1">
                <p className="text-gray-600">Water Depth Forecast</p>
                <p>{getWaterDepthLabel(alert.waterDepth)}</p>
              </div>
            </div>

            <div className="flex items-start gap-3">
              <AlertCircle className="w-5 h-5 text-gray-500 mt-0.5" />
              <div className="flex-1">
                <p className="text-gray-600">Source</p>
                <p>{alert.source}</p>
              </div>
            </div>

            <div className="flex items-start gap-3">
              <AlertCircle className="w-5 h-5 text-gray-500 mt-0.5" />
              <div className="flex-1">
                <p className="text-gray-600">Impact</p>
                <p>{alert.impact}</p>
              </div>
            </div>
          </div>
        </Card>

        {/* Safe Routes */}
        <Card className="p-4">
          <h3 className="mb-3">
            {mockSafeRoutes.length} Alternative Routes Available
          </h3>

          <div className="space-y-2">
            {mockSafeRoutes.map((route) => (
              <Card key={route.id} className="p-3 bg-gray-50">
                <div className="flex items-center justify-between mb-2">
                  <h4 className="text-sm">{route.name}</h4>
                  <Badge variant="secondary" className="text-xs">
                    {route.additionalTime}
                  </Badge>
                </div>
                <p className="text-xs text-gray-600 mb-2">{route.status}</p>
                <Button size="sm" className="w-full">
                  <Navigation className="w-4 h-4 mr-2" />
                  Navigate
                </Button>
              </Card>
            ))}
          </div>
        </Card>

        {/* Community Insights */}
        {reports.length > 0 && (
          <Card className="p-4">
            <h3 className="mb-3">
              Community Insights ({reports.length})
            </h3>

            <div className="space-y-3">
              {reports.map((report) => (
                <div key={report.id} className="flex gap-3">
                  <Avatar className="w-8 h-8">
                    <AvatarFallback>{report.userName.charAt(0)}</AvatarFallback>
                  </Avatar>
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="text-sm">{report.userName}</span>
                      <span className="text-xs text-gray-500">{report.timestamp}</span>
                      {report.verified && (
                        <Badge variant="secondary" className="text-xs">Verified</Badge>
                      )}
                    </div>
                    <p className="text-sm text-gray-600">
                      Water accumulating near underpass - {getWaterDepthLabel(report.waterDepth)}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </Card>
        )}

        {/* What Should I Do */}
        <Accordion type="single" collapsible className="bg-white rounded-lg">
          <AccordionItem value="item-1">
            <AccordionTrigger className="px-4">
              What Should I Do?
            </AccordionTrigger>
            <AccordionContent className="px-4 pb-4">
              <ul className="space-y-2 text-sm">
                <li className="flex items-start gap-2">
                  <span className="text-green-600">✓</span>
                  <span>Plan alternate route before {alert.expectedTime.split('-')[0].trim()}</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-green-600">✓</span>
                  <span>Avoid non-essential travel</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-green-600">✓</span>
                  <span>Monitor updates every 2 hours</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-green-600">✓</span>
                  <span>Secure ground-floor items</span>
                </li>
              </ul>
            </AccordionContent>
          </AccordionItem>
        </Accordion>
      </div>

      {/* Action Buttons */}
      <div className="fixed bottom-16 left-0 right-0 bg-white border-t p-4 space-y-2 safe-area-bottom">
        <Button className="w-full">
          <Navigation className="w-4 h-4 mr-2" />
          Navigate Safe Route
        </Button>
        <Button variant="outline" className="w-full">
          <Clock className="w-4 h-4 mr-2" />
          Set Reminder for 5:30 AM
        </Button>
      </div>
    </div>
  );
}
