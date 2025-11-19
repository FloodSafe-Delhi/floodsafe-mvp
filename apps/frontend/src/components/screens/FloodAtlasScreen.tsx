import MapComponent from '../MapComponent';
import { Badge } from '../ui/badge';

export function FloodAtlasScreen() {
    return (
        <div className="fixed inset-0 top-0 md:top-0 bottom-16 bg-transparent">
            {/* Full Screen Map */}
            <MapComponent className="w-full h-full" title="Flood Atlas" showControls={true} />
        </div>
    );
}
