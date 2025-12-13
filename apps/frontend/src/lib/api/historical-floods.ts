/**
 * API functions for historical flood data from IFI-Impacts.
 *
 * This module provides access to historical flood events from the
 * India Flood Inventory (IFI) dataset, which contains documented
 * flood events across India from 1967-2023.
 */
import { fetchJson } from './client';

export interface HistoricalFloodEvent {
  id: string;
  date: string;
  districts: string;  // Comma-separated district names
  severity: string;   // 'minor' | 'moderate' | 'severe'
  source: string;     // 'IFI-Impacts' or 'user_report'
  year: number;
  fatalities: number;
  injured: number;
  displaced: number;
  duration_days: number | null;
  main_cause: string;
  area_affected?: string;
}

export interface HistoricalFloodsResponse {
  type: 'FeatureCollection';
  features: Array<{
    type: 'Feature';
    geometry: {
      type: 'Point';
      coordinates: [number, number]; // [lng, lat]
    };
    properties: HistoricalFloodEvent;
  }>;
  metadata: {
    source: string;
    coverage: string;
    region: string;
    total_events: number;
    generated_at: string;
    message?: string;  // "Coming soon" message for unsupported cities
  };
}

/**
 * Fetch historical flood events for a city.
 *
 * @param city - City code (e.g., 'delhi', 'bangalore')
 * @returns GeoJSON FeatureCollection with historical flood events
 */
export async function getHistoricalFloods(city: string = 'delhi'): Promise<HistoricalFloodsResponse> {
  return fetchJson<HistoricalFloodsResponse>(`/historical-floods?city=${city}`);
}
