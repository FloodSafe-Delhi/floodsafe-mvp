import { FloodAlert } from '../../types';

export function AlertDetailScreen({ alert, onBack }: { alert: FloodAlert; onBack: () => void }) {
    return <div className="p-4">Alert Detail: {alert.location} <button onClick={onBack}>Back</button></div>;
}

export function AlertsListScreen({ onAlertClick }: { onAlertClick: (alert: FloodAlert) => void }) {
    return <div className="p-4">Alerts List</div>;
}
