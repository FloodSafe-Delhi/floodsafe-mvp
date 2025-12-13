/**
 * Centralized API configuration for FloodSafe frontend.
 * All API URL handling should import from this file.
 */

export const API_URL = import.meta.env.VITE_API_URL;

// Fail fast in production if API URL is not configured
if (!API_URL && import.meta.env.PROD) {
  throw new Error('VITE_API_URL environment variable is required in production');
}

// API base URL with /api suffix - use this for all API calls
export const API_BASE_URL = API_URL ? `${API_URL}/api` : 'http://localhost:8000/api';

// Raw API URL without /api suffix - use sparingly
export const API_ROOT_URL = API_URL || 'http://localhost:8000';
