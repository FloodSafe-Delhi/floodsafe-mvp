import React, { useState } from 'react';
import { useReportMutation } from '../lib/api/hooks';
import { X } from 'lucide-react';

interface ReportModalProps {
    isOpen: boolean;
    onClose: () => void;
    userLocation: { lat: number; lng: number } | null;
}

export const ReportModal: React.FC<ReportModalProps> = ({ isOpen, onClose, userLocation }) => {
    const [description, setDescription] = useState('');
    const [image, setImage] = useState<File | null>(null);
    const mutation = useReportMutation();

    if (!isOpen) return null;

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!userLocation) {
            alert("Location not available");
            return;
        }

        // TODO: Get actual user ID from auth context. Using a hardcoded ID for MVP demo if needed, 
        // or we should implement a simple login. For now, let's assume a demo user ID.
        const DEMO_USER_ID = "d53568ca-649e-4a59-92d4-135058513a91"; // Replace with one from your DB

        try {
            await mutation.mutateAsync({
                user_id: DEMO_USER_ID,
                description,
                latitude: userLocation.lat,
                longitude: userLocation.lng,
                image: image || undefined
            });
            alert("Report submitted successfully!");
            onClose();
            setDescription('');
            setImage(null);
        } catch (error) {
            alert("Failed to submit report");
            console.error(error);
        }
    };

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50">
            <div className="bg-white rounded-lg p-6 w-full max-w-md relative">
                <button onClick={onClose} className="absolute top-4 right-4 text-gray-500 hover:text-gray-700">
                    <X size={24} />
                </button>

                <h2 className="text-xl font-bold mb-4">Report Flood</h2>

                <form onSubmit={handleSubmit} className="space-y-4">
                    <div>
                        <label htmlFor="report-description" className="block text-sm font-medium text-gray-700">Description</label>
                        <textarea
                            id="report-description"
                            name="description"
                            className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 border p-2"
                            rows={3}
                            value={description}
                            onChange={(e) => setDescription(e.target.value)}
                            required
                        />
                    </div>

                    <div>
                        <label htmlFor="report-photo" className="block text-sm font-medium text-gray-700">Photo (Optional)</label>
                        <input
                            id="report-photo"
                            name="photo"
                            type="file"
                            accept="image/*"
                            onChange={(e) => setImage(e.target.files?.[0] || null)}
                            className="mt-1 block w-full"
                        />
                    </div>

                    <div className="text-sm text-gray-500">
                        Location: {userLocation ? `${userLocation.lat.toFixed(4)}, ${userLocation.lng.toFixed(4)}` : 'Locating...'}
                    </div>

                    <button
                        type="submit"
                        disabled={mutation.isPending || !userLocation}
                        className="w-full bg-blue-600 text-white py-2 px-4 rounded hover:bg-blue-700 disabled:bg-gray-400"
                    >
                        {mutation.isPending ? 'Submitting...' : 'Submit Report'}
                    </button>
                </form>
            </div>
        </div>
    );
};
