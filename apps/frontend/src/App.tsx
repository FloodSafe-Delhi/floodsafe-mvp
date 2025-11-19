import { MapComponent } from './components/MapComponent';

function App() {
    return (
        <div className="w-full h-screen">
            <MapComponent />
            <div className="absolute top-4 left-4 z-10 bg-white p-4 rounded shadow-lg">
                <h1 className="text-2xl font-bold mb-2">FloodSafe</h1>
                <p className="text-gray-600">Real-time flood monitoring</p>
            </div>
        </div>
    );
}

export default App;
