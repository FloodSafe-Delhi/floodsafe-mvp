import { useState, useRef } from 'react';
import { ArrowLeft, MapPin, Camera, Award, Phone, Shield } from 'lucide-react';
import { Card } from '../ui/card';
import { Button } from '../ui/button';
import { Progress } from '../ui/progress';
import { RadioGroup, RadioGroupItem } from '../ui/radio-group';
import { Label } from '../ui/label';
import { Input } from '../ui/input';
import { Checkbox } from '../ui/checkbox';
import { Badge } from '../ui/badge';
import { InputOTP, InputOTPGroup, InputOTPSlot } from '../ui/input-otp';
import { WaterDepth, VehiclePassability } from '../../types';
import { useReportMutation, useSendOTP, useVerifyOTP } from '../../lib/api/hooks';
import { toast } from 'sonner';

interface ReportScreenProps {
    onBack: () => void;
    onSubmit: () => void;
    onSelectLocation: () => void;
}

export function ReportScreen({ onBack, onSubmit, onSelectLocation }: ReportScreenProps) {
    const [step, setStep] = useState(1);
    const [waterDepth, setWaterDepth] = useState<WaterDepth>('knee');
    const [vehiclePassability, setVehiclePassability] = useState<VehiclePassability>('high-clearance');
    const [description, setDescription] = useState('');
    const [confirmed, setConfirmed] = useState(false);

    // New state for phone verification and camera
    const [phoneNumber, setPhoneNumber] = useState('');
    const [otp, setOtp] = useState('');
    const [verificationToken, setVerificationToken] = useState<string | null>(null);
    const [isPhoneVerified, setIsPhoneVerified] = useState(false);
    const [otpSent, setOtpSent] = useState(false);
    const [photoFile, setPhotoFile] = useState<File | null>(null);
    const [photoPreview, setPhotoPreview] = useState<string | null>(null);
    const [selectedLocation, setSelectedLocation] = useState<{lat: number, lng: number}>({
        lat: 12.9716, // Default Bangalore coordinates
        lng: 77.5946
    });

    const fileInputRef = useRef<HTMLInputElement>(null);

    const reportMutation = useReportMutation();
    const sendOTPMutation = useSendOTP();
    const verifyOTPMutation = useVerifyOTP();

    const totalSteps = 6; // Increased from 4 to 6
    const progressValue = (step / totalSteps) * 100;

    const handleNext = () => {
        // Validation before moving to next step
        if (step === 3 && !photoFile) {
            toast.error('Photo is required. Please take a photo of the flood.');
            return;
        }
        if (step === 4 && !phoneNumber) {
            toast.error('Phone number is required.');
            return;
        }
        if (step === 5 && !isPhoneVerified) {
            toast.error('Please verify your phone number first.');
            return;
        }

        if (step < totalSteps) {
            setStep(step + 1);
        } else {
            handleSubmit();
        }
    };

    const handleSendOTP = async () => {
        if (!phoneNumber || phoneNumber.length < 10) {
            toast.error('Please enter a valid phone number');
            return;
        }

        try {
            await sendOTPMutation.mutateAsync({ phone_number: phoneNumber });
            setOtpSent(true);
            toast.success('OTP sent to your phone!');
        } catch (error: any) {
            toast.error(error.message || 'Failed to send OTP');
        }
    };

    const handleVerifyOTP = async () => {
        if (otp.length !== 6) {
            toast.error('Please enter the 6-digit OTP');
            return;
        }

        try {
            const result = await verifyOTPMutation.mutateAsync({
                phone_number: phoneNumber,
                otp_code: otp,
            });

            if (result.verified && result.token) {
                setVerificationToken(result.token);
                setIsPhoneVerified(true);
                toast.success('Phone verified successfully!');
                setStep(6); // Move to confirmation step
            } else {
                toast.error(result.message || 'Invalid OTP');
            }
        } catch (error: any) {
            toast.error(error.message || 'Verification failed');
        }
    };

    const handlePhotoCapture = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file) {
            // Validate file type
            if (!file.type.startsWith('image/')) {
                toast.error('Please select an image file');
                return;
            }

            // Validate file size (max 10MB)
            if (file.size > 10 * 1024 * 1024) {
                toast.error('Image must be less than 10MB');
                return;
            }

            setPhotoFile(file);

            // Generate preview
            const reader = new FileReader();
            reader.onload = () => setPhotoPreview(reader.result as string);
            reader.readAsDataURL(file);

            toast.success('Photo captured! GPS data will be extracted automatically.');
        }
    };

    const handleSubmit = async () => {
        if (!photoFile) {
            toast.error('Photo is required');
            return;
        }

        if (!verificationToken) {
            toast.error('Phone verification is required');
            return;
        }

        try {
            await reportMutation.mutateAsync({
                latitude: selectedLocation.lat,
                longitude: selectedLocation.lng,
                description: description || `Water depth: ${waterDepth}, Vehicle: ${vehiclePassability}`,
                user_id: "d53568ca-649e-4a59-92d4-135058513a91", // TODO: Use actual auth
                phone_number: phoneNumber,
                phone_verification_token: verificationToken,
                water_depth: waterDepth,
                vehicle_passability: vehiclePassability,
                image: photoFile,
            });

            toast.success('Report submitted successfully! +15 points', {
                description: 'Your report is being validated against IoT sensors.',
            });
            onSubmit();
        } catch (error: any) {
            toast.error('Failed to submit report', {
                description: error.message || 'Please try again',
            });
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
                        Up to +15 pts
                    </Badge>
                </div>
                <Progress value={progressValue} className="h-2" />
                <div className="grid grid-cols-6 gap-1 text-[10px] text-gray-500 mt-2">
                    <span className={step >= 1 ? 'text-blue-600 font-medium' : ''}>Location</span>
                    <span className={step >= 2 ? 'text-blue-600 font-medium' : ''}>Details</span>
                    <span className={step >= 3 ? 'text-blue-600 font-medium' : ''}>Photo</span>
                    <span className={step >= 4 ? 'text-blue-600 font-medium' : ''}>Phone</span>
                    <span className={step >= 5 ? 'text-blue-600 font-medium' : ''}>Verify</span>
                    <span className={step >= 6 ? 'text-blue-600 font-medium' : ''}>Confirm</span>
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

                            <Button variant="outline" className="w-full" onClick={onSelectLocation}>
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

                {/* Step 3: Photo (MANDATORY) */}
                {step === 3 && (
                    <Card className="p-4">
                        <div className="flex items-center gap-2 mb-2">
                            <h3>Add Photo</h3>
                            <Badge variant="destructive" className="text-xs">Required</Badge>
                        </div>
                        <p className="text-sm text-gray-600 mb-4">
                            Photo with GPS coordinates is mandatory for validation
                        </p>

                        {!photoPreview ? (
                            <div className="border-2 border-dashed border-blue-300 rounded-lg p-8 bg-blue-50">
                                <div className="text-center">
                                    <Camera className="w-12 h-12 mx-auto mb-3 text-blue-600" />
                                    <input
                                        ref={fileInputRef}
                                        type="file"
                                        accept="image/*"
                                        capture="environment"
                                        onChange={handlePhotoCapture}
                                        className="hidden"
                                        id="camera-input"
                                    />
                                    <label htmlFor="camera-input">
                                        <Button type="button" variant="default" className="mb-2" asChild>
                                            <span className="cursor-pointer">
                                                <Camera className="w-4 h-4 mr-2" />
                                                Take Photo
                                            </span>
                                        </Button>
                                    </label>
                                    <p className="text-xs text-gray-600 mt-2">
                                        Make sure location services are enabled
                                    </p>
                                </div>
                            </div>
                        ) : (
                            <div className="space-y-3">
                                <div className="relative rounded-lg overflow-hidden border-2 border-green-500">
                                    <img
                                        src={photoPreview}
                                        alt="Report preview"
                                        className="w-full h-auto"
                                    />
                                    <div className="absolute top-2 right-2">
                                        <Badge className="bg-green-500">
                                            ✓ Photo captured
                                        </Badge>
                                    </div>
                                </div>
                                <Button
                                    variant="outline"
                                    className="w-full"
                                    onClick={() => {
                                        setPhotoFile(null);
                                        setPhotoPreview(null);
                                    }}
                                >
                                    Retake Photo
                                </Button>
                                <div className="bg-blue-50 border border-blue-200 rounded-lg p-3 text-xs">
                                    <p className="text-blue-900">
                                        📍 GPS coordinates will be automatically extracted from photo metadata
                                    </p>
                                </div>
                            </div>
                        )}
                    </Card>
                )}

                {/* Step 4: Phone Number */}
                {step === 4 && (
                    <Card className="p-4">
                        <div className="flex items-center gap-2 mb-2">
                            <Phone className="w-5 h-5 text-blue-600" />
                            <h3>Verify Phone Number</h3>
                        </div>
                        <p className="text-sm text-gray-600 mb-4">
                            We'll send a verification code to confirm your identity
                        </p>

                        <div className="space-y-4">
                            <div>
                                <Label htmlFor="phone">Phone Number</Label>
                                <div className="flex gap-2 mt-2">
                                    <Input
                                        id="phone"
                                        type="tel"
                                        placeholder="+91 98765 43210"
                                        value={phoneNumber}
                                        onChange={(e) => setPhoneNumber(e.target.value)}
                                        disabled={otpSent}
                                        className="flex-1"
                                    />
                                    <Button
                                        onClick={handleSendOTP}
                                        disabled={otpSent || sendOTPMutation.isPending || !phoneNumber}
                                        className="min-w-[100px]"
                                    >
                                        {sendOTPMutation.isPending ? 'Sending...' : otpSent ? 'Sent ✓' : 'Send OTP'}
                                    </Button>
                                </div>
                            </div>

                            {otpSent && (
                                <div className="bg-green-50 border border-green-200 rounded-lg p-3">
                                    <p className="text-sm text-green-900">
                                        ✓ OTP sent to {phoneNumber}
                                    </p>
                                    <p className="text-xs text-green-700 mt-1">
                                        Check your messages for the 6-digit code
                                    </p>
                                </div>
                            )}

                            <div className="text-xs text-gray-500 space-y-1">
                                <p>• OTP expires in 5 minutes</p>
                                <p>• Maximum 3 requests per hour</p>
                            </div>
                        </div>
                    </Card>
                )}

                {/* Step 5: OTP Verification */}
                {step === 5 && (
                    <Card className="p-4">
                        <div className="flex items-center gap-2 mb-2">
                            <Shield className="w-5 h-5 text-green-600" />
                            <h3>Enter Verification Code</h3>
                        </div>
                        <p className="text-sm text-gray-600 mb-4">
                            Enter the 6-digit code sent to {phoneNumber}
                        </p>

                        <div className="space-y-4">
                            <div className="flex justify-center">
                                <InputOTP
                                    maxLength={6}
                                    value={otp}
                                    onChange={setOtp}
                                >
                                    <InputOTPGroup>
                                        <InputOTPSlot index={0} />
                                        <InputOTPSlot index={1} />
                                        <InputOTPSlot index={2} />
                                        <InputOTPSlot index={3} />
                                        <InputOTPSlot index={4} />
                                        <InputOTPSlot index={5} />
                                    </InputOTPGroup>
                                </InputOTP>
                            </div>

                            <Button
                                className="w-full"
                                onClick={handleVerifyOTP}
                                disabled={otp.length !== 6 || verifyOTPMutation.isPending}
                            >
                                {verifyOTPMutation.isPending ? 'Verifying...' : 'Verify Code'}
                            </Button>

                            <div className="text-center">
                                <Button
                                    variant="link"
                                    size="sm"
                                    onClick={() => {
                                        setOtpSent(false);
                                        setOtp('');
                                        setStep(4);
                                    }}
                                    className="text-xs"
                                >
                                    Didn't receive code? Try again
                                </Button>
                            </div>
                        </div>
                    </Card>
                )}

                {/* Step 6: Confirmation */}
                {step === 6 && (
                    <div className="space-y-4">
                        <Card className="p-4">
                            <h3 className="mb-4">Report Summary</h3>

                            <div className="space-y-3 text-sm">
                                <div>
                                    <p className="text-gray-600 text-xs">Location</p>
                                    <p className="font-medium">
                                        {selectedLocation.lat.toFixed(4)}, {selectedLocation.lng.toFixed(4)}
                                    </p>
                                    <p className="text-xs text-gray-500">Bangalore, Karnataka</p>
                                </div>
                                <div className="h-px bg-gray-200" />
                                <div>
                                    <p className="text-gray-600 text-xs">Water Depth</p>
                                    <p className="font-medium capitalize">{waterDepth}</p>
                                </div>
                                <div className="h-px bg-gray-200" />
                                <div>
                                    <p className="text-gray-600 text-xs">Vehicle Passability</p>
                                    <p className="font-medium capitalize">{vehiclePassability.replace('-', ' ')}</p>
                                </div>
                                <div className="h-px bg-gray-200" />
                                <div>
                                    <p className="text-gray-600 text-xs">Photo</p>
                                    <div className="flex items-center gap-2 mt-1">
                                        <Badge variant="secondary" className="text-xs">
                                            ✓ With GPS
                                        </Badge>
                                        {photoFile && (
                                            <span className="text-xs text-gray-500">
                                                {(photoFile.size / 1024 / 1024).toFixed(2)} MB
                                            </span>
                                        )}
                                    </div>
                                </div>
                                <div className="h-px bg-gray-200" />
                                <div>
                                    <p className="text-gray-600 text-xs">Phone Number</p>
                                    <div className="flex items-center gap-2 mt-1">
                                        <p className="font-medium">{phoneNumber}</p>
                                        <Badge className="bg-green-500 text-xs">
                                            ✓ Verified
                                        </Badge>
                                    </div>
                                </div>
                            </div>
                        </Card>

                        <Card className="p-4">
                            <h3 className="mb-4">Confirm & Submit</h3>

                            <div className="space-y-4">
                                <div className="flex items-start gap-3">
                                    <Checkbox
                                        id="confirm"
                                        checked={confirmed}
                                        onCheckedChange={(checked) => setConfirmed(checked as boolean)}
                                    />
                                    <Label htmlFor="confirm" className="text-sm cursor-pointer leading-relaxed">
                                        I confirm this report is accurate and taken at the actual flood location
                                    </Label>
                                </div>

                                <div className="p-3 bg-blue-50 border border-blue-200 rounded-lg space-y-2">
                                    <p className="text-xs text-blue-900 font-medium">
                                        What happens next:
                                    </p>
                                    <ul className="text-xs text-blue-800 space-y-1 ml-4 list-disc">
                                        <li>Your report will be validated against nearby IoT sensors</li>
                                        <li>High-confidence reports are auto-verified (+15 points!)</li>
                                        <li>Your location is anonymized to 100m radius for privacy</li>
                                    </ul>
                                </div>

                                <div className="p-3 bg-gray-50 rounded-lg">
                                    <p className="text-xs text-gray-600">
                                        🔒 Privacy: Phone number is never shared publicly
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
