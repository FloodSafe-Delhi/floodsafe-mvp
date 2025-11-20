import MapComponent from '../MapComponent';

export function FloodAtlasScreen() {
    return (
        <div className="fixed inset-0 top-14 md:top-0 bottom-16 bg-transparent">
            <MapComponent
                className="w-full h-full"
                title="Flood Atlas"
                showControls={true}
                showCitySelector={true}
            />
        </div>
    );
}
