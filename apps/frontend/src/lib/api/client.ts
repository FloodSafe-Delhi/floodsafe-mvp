import { TokenStorage } from '../auth/token-storage';
import { API_BASE_URL } from './config';

/**
 * Get authorization headers if user is authenticated.
 */
function getAuthHeaders(): Record<string, string> {
    const token = TokenStorage.getAccessToken();
    if (token) {
        return { 'Authorization': `Bearer ${token}` };
    }
    return {};
}

/**
 * Handle 401 responses by attempting token refresh.
 * Returns true if refresh succeeded and request should be retried.
 */
async function handleUnauthorized(): Promise<boolean> {
    const refreshToken = TokenStorage.getRefreshToken();
    if (!refreshToken) {
        return false;
    }

    try {
        const response = await fetch(`${API_BASE_URL}/auth/refresh`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ refresh_token: refreshToken }),
        });

        if (!response.ok) {
            TokenStorage.clearTokens();
            return false;
        }

        const tokens = await response.json();
        TokenStorage.setTokens(tokens.access_token, tokens.refresh_token);
        return true;
    } catch {
        TokenStorage.clearTokens();
        return false;
    }
}

export async function fetchJson<T>(endpoint: string, options?: RequestInit): Promise<T> {
    const makeRequest = async () => {
        const response = await fetch(`${API_BASE_URL}${endpoint}`, {
            ...options,
            headers: {
                'Content-Type': 'application/json',
                ...getAuthHeaders(),
                ...options?.headers,
            },
        });

        return response;
    };

    let response = await makeRequest();

    // Handle 401 by refreshing token and retrying
    if (response.status === 401) {
        const refreshed = await handleUnauthorized();
        if (refreshed) {
            response = await makeRequest();
        }
    }

    if (!response.ok) {
        // Try to get detailed error message
        let errorMessage = `API Error: ${response.statusText}`;
        try {
            const errorData = await response.json();
            if (errorData.detail) {
                errorMessage = typeof errorData.detail === 'string'
                    ? errorData.detail
                    : JSON.stringify(errorData.detail);
            }
        } catch {
            // Ignore JSON parse errors
        }
        throw new Error(errorMessage);
    }

    return response.json();
}

export async function uploadFile<T>(endpoint: string, formData: FormData): Promise<T> {
    const makeRequest = async () => {
        const response = await fetch(`${API_BASE_URL}${endpoint}`, {
            method: 'POST',
            headers: {
                ...getAuthHeaders(),
            },
            body: formData,
        });

        return response;
    };

    let response = await makeRequest();

    // Handle 401 by refreshing token and retrying
    if (response.status === 401) {
        const refreshed = await handleUnauthorized();
        if (refreshed) {
            response = await makeRequest();
        }
    }

    if (!response.ok) {
        // Try to get detailed error message from response body
        let errorMessage = `API Error: ${response.statusText}`;
        let errorDetails = null;
        try {
            const responseText = await response.text();
            console.error('Upload failed - Raw response:', responseText);
            try {
                errorDetails = JSON.parse(responseText);
                if (errorDetails.detail) {
                    errorMessage = typeof errorDetails.detail === 'string'
                        ? errorDetails.detail
                        : JSON.stringify(errorDetails.detail);
                }
            } catch {
                // Not JSON, use text as-is
                if (responseText) {
                    errorMessage = responseText;
                }
            }
        } catch (e) {
            console.error('Failed to read error response:', e);
        }
        console.error('Upload failed with status:', response.status, errorMessage);
        throw new Error(errorMessage);
    }

    return response.json();
}
