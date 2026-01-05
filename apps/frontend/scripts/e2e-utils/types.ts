/**
 * Database record types for E2E test assertions.
 * These match the Supabase schema and are used for verification.
 */

export interface UserRecord {
  id: string;
  email: string | null;
  username: string | null;
  role: string | null;
  auth_provider: string | null;
  profile_complete: boolean | null;
  city_preference: string | null;
  onboarding_step: number | null;
  points: number | null;
  level: number | null;
  reports_count: number | null;
  created_at: string | null;
}

export interface ReportRecord {
  id: string;
  user_id: string | null;
  description: string | null;
  media_url: string | null;
  media_type: string | null;
  verified: boolean | null;
  location_verified: boolean | null;
  water_depth: string | null;
  vehicle_passability: string | null;
  upvotes: number | null;
  downvotes: number | null;
  timestamp: string | null;
}

export interface WatchAreaRecord {
  id: string;
  user_id: string | null;
  name: string | null;
  radius: number | null;
  created_at: string | null;
}

export interface CommentRecord {
  id: string;
  report_id: string;
  user_id: string;
  content: string;
  created_at: string | null;
}

export interface VoteRecord {
  id: string;
  user_id: string;
  report_id: string;
  vote_type: string;
  created_at: string | null;
}

export interface AlertRecord {
  id: string;
  user_id: string;
  report_id: string;
  watch_area_id: string;
  message: string;
  is_read: boolean | null;
  created_at: string | null;
}

/**
 * Partial types for assertion matching.
 * Use these when verifying specific fields.
 */
export type PartialUser = Partial<Omit<UserRecord, 'id'>>;
export type PartialReport = Partial<Omit<ReportRecord, 'id'>>;
export type PartialWatchArea = Partial<Omit<WatchAreaRecord, 'id'>>;
