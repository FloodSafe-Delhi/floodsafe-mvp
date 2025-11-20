import MapComponent from '../MapComponent';
import { Badge } from '../ui/badge';

interface FloodAtlasScreenProps {
    cityCode?: 'BLR' | 'DEL';
}

export function FloodAtlasScreen({ cityCode }: FloodAtlasScreenProps) {
    return (
        <div className="w-full h-screen relative">
            {/* Full Screen Map */}
            <MapComponent className="w-full h-full" cityCode={cityCode} />

            {/* Offline Indicator */}
            <div className="absolute top-4 right-4 pointer-events-none z-10">
                <Badge variant="secondary" className="bg-white shadow">
                    Online
                </Badge>
            </div>

            {/* Optional: Add a back button for mobile */}
            <div className="md:hidden absolute top-4 left-4 z-10">
                <Badge variant="secondary" className="bg-white shadow">
                    Flood Atlas
                </Badge>
            </div>
        </div>
    );
}
