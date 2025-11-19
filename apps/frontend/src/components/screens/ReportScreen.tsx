import { useState } from 'react';
import { ArrowLeft, MapPin, Camera, Award } from 'lucide-react';
import { Card } from '../ui/card';
import { Button } from '../ui/button';
import { Progress } from '../ui/progress';
import { RadioGroup, RadioGroupItem } from '../ui/radio-group';
import { Label } from '../ui/label';
import { Input } from '../ui/input';
import { Checkbox } from '../ui/checkbox';
import { Badge } from '../ui/badge';
import { WaterDepth, VehiclePassability } from '../../types';
import { useReportMutation } from '../../lib/api/hooks';
import { toast } from 'sonner';

interface ReportScreenProps {
    onBack: () => void;
    onSubmit: () => void;
}

export function ReportScreen({ onBack, onSubmit }: ReportScreenProps) {
    const [step, setStep] = useState(1);
    const [waterDepth, setWaterDepth] = useState<WaterDepth>('knee');
    const [vehiclePassability, setVehiclePassability] = useState<VehiclePassability>('high-clearance');
    const [description, setDescription] = useState('');
    const [confirmed, setConfirmed] = useState(false);

    const reportMutation = useReportMutation();

    const totalSteps = 4;
    const progressValue = (step / totalSteps) * 100;

    const handleNext = () => {
        if (step < totalSteps) {
            setStep(step + 1);
        } else {
            handleSubmit();
        }
    };

    const handleSubmit = async () => {
        try {
            // Hardcoded location for now, ideally we get this from a map picker in Step 1
            await reportMutation.mutateAsync({
                latitude: 12.9716,
                longitude: 77.5946,
                description: `${description} - Depth: ${waterDepth}, Passability: ${vehiclePassability}`,
                image: null // Image upload not implemented in UI yet
            });
            toast.success('Report submitted successfully!');
            onSubmit();
        } catch (error) {
            toast.error('Failed to submit report.');
            console.error(error);
        }
    };

    const handleBack = () => {
        if (step > 1) {
            setStep(step - 1);
        } else {
            onBack();
        }
    };

    return (
        <div className="pb-16 min-h-screen bg-gray-50">
            {/* Header */}
            <div className="bg-white shadow-sm sticky top-14 z-40">
                <div className="flex items-center justify-between px-4 h-14">
                    <button
                        onClick={handleBack}
                        className="p-2 -ml-2 min-w-[44px] min-h-[44px] flex items-center justify-center"
                        aria-label="Go back"
                    >
                        <ArrowLeft className="w-6 h-6" />
                    </button>

                    <h1 className="flex-1 text-center">
                        Report Flooding
                    </h1>

                    <div className="w-10"></div>
                </div>
            </div>

            {/* Progress Indicator */}
            <div className="bg-white px-4 pb-4">
                <div className="flex items-center justify-between mb-2">
                    <span className="text-sm">Step {step} of {totalSteps}</span>
                    <Badge variant="secondary" className="text-xs">
                        <Award className="w-3 h-3 mr-1" />
                        Score: +10 pts
                    </Badge>
                </div>
                <Progress value={progressValue} className="h-2" />
                <div className="flex justify-between text-xs text-gray-500 mt-2">
                    <span className={step >= 1 ? 'text-blue-600' : ''}>Location</span>
                    <span className={step >= 2 ? 'text-blue-600' : ''}>Details</span>
                    <span className={step >= 3 ? 'text-blue-600' : ''}>Photo</span>
                    <span className={step >= 4 ? 'text-blue-600' : ''}>Confirm</span>
                </div>
            </div>

            {/* Form Content */}
            <div className="p-4 space-y-4">
                {/* Step 1: Location */}
                {step === 1 && (
                    <Card className="p-4">
                        <h3 className="mb-4">Select Location</h3>

                        <div className="space-y-4">
                            <div className="bg-green-50 border border-green-200 rounded-lg p-3">
                                <div className="flex items-start gap-2">
                                    <MapPin className="w-5 h-5 text-green-600 mt-0.5" />
                                    <div className="flex-1">
                                        <p className="text-sm">Current Location</p>
                                        <p className="text-xs text-gray-600">Bangalore (Simulated)</p>
                                        <p className="text-xs text-gray-500 mt-1">GPS accuracy: ±5m</p>
                                    </div>
                                </div>
                            </div>

                            <div className="text-center text-gray-500 text-sm">OR</div>

                            <Button variant="outline" className="w-full">
                                <MapPin className="w-4 h-4 mr-2" />
                                Select from Map
                            </Button>

                            <div>
                                <Label htmlFor="desc">Description</Label>
                                <Input
                                    id="desc"
                                    placeholder="Describe the situation..."
                                    className="mt-1"
                                    value={description}
                                    onChange={(e) => setDescription(e.target.value)}
                                />
                            </div>
                        </div>
                    </Card>
                )}

                {/* Step 2: Flood Details */}
                {step === 2 && (
                    <Card className="p-4 space-y-6">
                        <div>
                            <h3 className="mb-4">Water Depth</h3>
                            <RadioGroup value={waterDepth} onValueChange={(v) => setWaterDepth(v as WaterDepth)}>
                                <div className="space-y-3">
                                    {[
                                        { value: 'ankle', label: 'Ankle-deep', sublabel: '< 0.3m', emoji: '🚶' },
                                        { value: 'knee', label: 'Knee-deep', sublabel: '0.3-0.6m', emoji: '🚶‍♂️' },
                                        { value: 'waist', label: 'Waist-deep', sublabel: '0.6-1.2m', emoji: '🏊' },
                                        { value: 'impassable', label: 'Impassable', sublabel: '> 1.2m', emoji: '⚠️' }
                                    ].map((option) => (
                                        <div key={option.value} className="flex items-center space-x-3 p-3 border rounded-lg hover:bg-gray-50">
                                            <RadioGroupItem value={option.value} id={option.value} />
                                            <Label htmlFor={option.value} className="flex-1 flex items-center gap-3 cursor-pointer">
                                                <span className="text-2xl">{option.emoji}</span>
                                                <div>
                                                    <p className="text-sm">{option.label}</p>
                                                    <p className="text-xs text-gray-500">{option.sublabel}</p>
                                                </div>
                                            </Label>
                                        </div>
                                    ))}
                                </div>
                            </RadioGroup>
                        </div>

                        <div>
                            <h3 className="mb-4">Vehicle Passability</h3>
                            <RadioGroup value={vehiclePassability} onValueChange={(v) => setVehiclePassability(v as VehiclePassability)}>
                                <div className="space-y-3">
                                    {[
                                        { value: 'all', label: 'All vehicles passing', icon: '🚗' },
                                        { value: 'high-clearance', label: 'High-clearance only', sublabel: 'SUVs, buses', icon: '🚙' },
                                        { value: 'none', label: 'No vehicles passing', icon: '🚫' }
                                    ].map((option) => (
                                        <div key={option.value} className="flex items-center space-x-3 p-3 border rounded-lg hover:bg-gray-50">
                                            <RadioGroupItem value={option.value} id={`vehicle-${option.value}`} />
                                            <Label htmlFor={`vehicle-${option.value}`} className="flex-1 flex items-center gap-3 cursor-pointer">
                                                <span className="text-xl">{option.icon}</span>
                                                <div>
                                                    <p className="text-sm">{option.label}</p>
                                                    {option.sublabel && <p className="text-xs text-gray-500">{option.sublabel}</p>}
                                                </div>
                                            </Label>
                                        </div>
                                    ))}
                                </div>
                            </RadioGroup>
                        </div>
                    </Card>
                )}

                {/* Step 3: Photo */}
                {step === 3 && (
                    <Card className="p-4">
                        <h3 className="mb-2">Add Photo (Optional)</h3>
                        <p className="text-sm text-gray-600 mb-4">Photos help validate reports faster</p>

                        <div className="border-2 border-dashed border-gray-300 rounded-lg p-8">
                            <div className="text-center">
                                <Camera className="w-12 h-12 mx-auto mb-3 text-gray-400" />
                                <Button variant="outline" className="mb-2">
                                    <Camera className="w-4 h-4 mr-2" />
                                    Take Photo
                                </Button>
                                <p className="text-xs text-gray-500">or choose from gallery</p>
                            </div>
                        </div>
                    </Card>
                )}

                {/* Step 4: Confirmation */}
                {step === 4 && (
                    <div className="space-y-4">
                        <Card className="p-4">
                            <h3 className="mb-4">Report Summary</h3>

                            <div className="space-y-3 text-sm">
                                <div>
                                    <p className="text-gray-600">Location</p>
                                    <p>Bangalore (Simulated)</p>
                                </div>
                                <div>
                                    <p className="text-gray-600">Water Depth</p>
                                    <p>{waterDepth}</p>
                                </div>
                                <div>
                                    <p className="text-gray-600">Vehicle Passability</p>
                                    <p>{vehiclePassability}</p>
                                </div>
                            </div>
                        </Card>

                        <Card className="p-4">
                            <h3 className="mb-4">Verify Your Report</h3>

                            <div className="space-y-4">
                                <div className="flex items-start gap-2">
                                    <Checkbox
                                        id="confirm"
                                        checked={confirmed}
                                        onCheckedChange={(checked) => setConfirmed(checked as boolean)}
                                    />
                                    <Label htmlFor="confirm" className="text-sm cursor-pointer">
                                        I confirm this report is accurate
                                    </Label>
                                </div>

                                <div className="p-3 bg-gray-50 rounded-lg">
                                    <p className="text-xs text-gray-600">
                                        🔒 Privacy: Location anonymized to 100m radius.
                                    </p>
                                </div>
                            </div>
                        </Card>
                    </div>
                )}
            </div>

            {/* Action Buttons */}
            <div className="fixed bottom-16 left-0 right-0 bg-white border-t p-4 space-y-2 safe-area-bottom">
                {step < totalSteps ? (
                    <Button className="w-full" onClick={handleNext}>
                        Continue
                    </Button>
                ) : (
                    <Button
                        className="w-full"
                        onClick={handleNext}
                        disabled={!confirmed || reportMutation.isPending}
                    >
                        {reportMutation.isPending ? 'Submitting...' : 'Submit Report'}
                    </Button>
                )}
                {step > 1 && (
                    <Button variant="ghost" className="w-full" onClick={handleBack}>
                        Back
                    </Button>
                )}
            </div>
        </div>
    );
}
