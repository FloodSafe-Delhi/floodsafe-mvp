/**
 * Database Assertion Utilities for E2E Tests
 *
 * Uses Supabase REST API with service role key for direct database access.
 * This bypasses RLS to verify data persistence in tests.
 *
 * SECURITY: Service role key should ONLY be used in test scripts, never in production code.
 */

import type {
  UserRecord,
  ReportRecord,
  WatchAreaRecord,
  CommentRecord,
  VoteRecord,
  PartialUser,
  PartialReport,
} from './types';

// Environment configuration
const SUPABASE_URL = process.env.SUPABASE_URL || 'https://udblirsscaghsepuxxqv.supabase.co';
const SUPABASE_SERVICE_ROLE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY;

if (!SUPABASE_SERVICE_ROLE_KEY) {
  console.warn('⚠️  SUPABASE_SERVICE_ROLE_KEY not set - database assertions will fail');
}

/**
 * Make a request to Supabase REST API.
 */
async function supabaseQuery<T>(
  table: string,
  query: string = '',
  method: 'GET' | 'DELETE' = 'GET'
): Promise<T[]> {
  if (!SUPABASE_SERVICE_ROLE_KEY) {
    throw new Error('SUPABASE_SERVICE_ROLE_KEY is required for database assertions');
  }

  const url = `${SUPABASE_URL}/rest/v1/${table}${query}`;

  const response = await fetch(url, {
    method,
    headers: {
      'apikey': SUPABASE_SERVICE_ROLE_KEY,
      'Authorization': `Bearer ${SUPABASE_SERVICE_ROLE_KEY}`,
      'Content-Type': 'application/json',
      'Prefer': method === 'DELETE' ? 'return=minimal' : 'return=representation',
    },
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`Supabase query failed: ${response.status} - ${error}`);
  }

  if (method === 'DELETE') {
    return [] as T[];
  }

  return response.json();
}

/**
 * Database assertion utilities class.
 */
export class DbAssertions {
  private testEmail: string;
  private testUserId: string | null = null;

  constructor(testEmail: string) {
    this.testEmail = testEmail;
  }

  /**
   * Verify a user was created in the database.
   */
  async verifyUserCreated(email?: string): Promise<UserRecord | null> {
    const targetEmail = email || this.testEmail;
    const users = await supabaseQuery<UserRecord>(
      'users',
      `?email=eq.${encodeURIComponent(targetEmail)}&select=*`
    );

    if (users.length > 0) {
      this.testUserId = users[0].id;
      return users[0];
    }
    return null;
  }

  /**
   * Verify user profile was updated with expected fields.
   */
  async verifyUserUpdated(
    userId: string,
    expectedFields: PartialUser
  ): Promise<boolean> {
    const users = await supabaseQuery<UserRecord>(
      'users',
      `?id=eq.${userId}&select=*`
    );

    if (users.length === 0) {
      return false;
    }

    const user = users[0];

    // Check each expected field matches
    for (const [key, expectedValue] of Object.entries(expectedFields)) {
      const actualValue = user[key as keyof UserRecord];
      if (actualValue !== expectedValue) {
        console.log(`  [DB] Field mismatch: ${key} - expected "${expectedValue}", got "${actualValue}"`);
        return false;
      }
    }

    return true;
  }

  /**
   * Verify a report was created with expected fields.
   */
  async verifyReportCreated(
    userId: string,
    expectedFields?: PartialReport
  ): Promise<ReportRecord | null> {
    // Get most recent report by this user
    const reports = await supabaseQuery<ReportRecord>(
      'reports',
      `?user_id=eq.${userId}&select=*&order=timestamp.desc&limit=1`
    );

    if (reports.length === 0) {
      return null;
    }

    const report = reports[0];

    // If expected fields provided, verify they match
    if (expectedFields) {
      for (const [key, expectedValue] of Object.entries(expectedFields)) {
        const actualValue = report[key as keyof ReportRecord];
        if (actualValue !== expectedValue) {
          console.log(`  [DB] Report field mismatch: ${key} - expected "${expectedValue}", got "${actualValue}"`);
        }
      }
    }

    return report;
  }

  /**
   * Verify a watch area was created.
   */
  async verifyWatchAreaCreated(
    userId: string,
    name?: string
  ): Promise<WatchAreaRecord | null> {
    let query = `?user_id=eq.${userId}&select=*&order=created_at.desc&limit=1`;

    if (name) {
      query = `?user_id=eq.${userId}&name=ilike.${encodeURIComponent(`%${name}%`)}&select=*&order=created_at.desc&limit=1`;
    }

    const areas = await supabaseQuery<WatchAreaRecord>('watch_areas', query);
    return areas.length > 0 ? areas[0] : null;
  }

  /**
   * Verify a vote was recorded.
   */
  async verifyVoteRecorded(
    userId: string,
    reportId: string,
    voteType: 'upvote' | 'downvote'
  ): Promise<boolean> {
    const votes = await supabaseQuery<VoteRecord>(
      'report_votes',
      `?user_id=eq.${userId}&report_id=eq.${reportId}&vote_type=eq.${voteType}&select=*`
    );
    return votes.length > 0;
  }

  /**
   * Verify a comment was created.
   */
  async verifyCommentCreated(
    reportId: string,
    contentSubstring: string
  ): Promise<CommentRecord | null> {
    const comments = await supabaseQuery<CommentRecord>(
      'comments',
      `?report_id=eq.${reportId}&content=ilike.${encodeURIComponent(`%${contentSubstring}%`)}&select=*`
    );
    return comments.length > 0 ? comments[0] : null;
  }

  /**
   * Get count of records in a table for a user.
   */
  async getRecordCount(table: string, userId: string): Promise<number> {
    const records = await supabaseQuery<{ id: string }>(
      table,
      `?user_id=eq.${userId}&select=id`
    );
    return records.length;
  }

  /**
   * Clean up all test data created during E2E run.
   * Deletes in correct order to respect foreign key constraints.
   */
  async cleanup(): Promise<void> {
    if (!this.testUserId) {
      // Try to find the user by email
      const user = await this.verifyUserCreated();
      if (!user) {
        console.log('  [Cleanup] No test user found to clean up');
        return;
      }
    }

    const userId = this.testUserId!;
    console.log(`  [Cleanup] Deleting test data for user: ${userId}`);

    try {
      // Delete in order respecting foreign keys
      // 1. Delete alerts (references reports and watch_areas)
      await supabaseQuery('alerts', `?user_id=eq.${userId}`, 'DELETE');
      console.log('    - Deleted alerts');

      // 2. Delete report_votes
      await supabaseQuery('report_votes', `?user_id=eq.${userId}`, 'DELETE');
      console.log('    - Deleted report_votes');

      // 3. Delete comments
      await supabaseQuery('comments', `?user_id=eq.${userId}`, 'DELETE');
      console.log('    - Deleted comments');

      // 4. Delete reports
      await supabaseQuery('reports', `?user_id=eq.${userId}`, 'DELETE');
      console.log('    - Deleted reports');

      // 5. Delete watch_areas
      await supabaseQuery('watch_areas', `?user_id=eq.${userId}`, 'DELETE');
      console.log('    - Deleted watch_areas');

      // 6. Delete refresh_tokens
      await supabaseQuery('refresh_tokens', `?user_id=eq.${userId}`, 'DELETE');
      console.log('    - Deleted refresh_tokens');

      // 7. Delete email_verification_tokens
      await supabaseQuery('email_verification_tokens', `?user_id=eq.${userId}`, 'DELETE');
      console.log('    - Deleted email_verification_tokens');

      // 8. Delete reputation_history
      await supabaseQuery('reputation_history', `?user_id=eq.${userId}`, 'DELETE');
      console.log('    - Deleted reputation_history');

      // 9. Finally delete the user
      await supabaseQuery('users', `?id=eq.${userId}`, 'DELETE');
      console.log('    - Deleted user');

      console.log('  [Cleanup] Complete!');
    } catch (error) {
      console.error('  [Cleanup] Error during cleanup:', error);
      throw error;
    }
  }
}

/**
 * Create a new DbAssertions instance for a test email.
 */
export function createDbAssertions(testEmail: string): DbAssertions {
  return new DbAssertions(testEmail);
}

/**
 * Log a successful database assertion.
 */
export function logDbSuccess(message: string): void {
  console.log(`  ✓ [DB] ${message}`);
}

/**
 * Log a failed database assertion.
 */
export function logDbFailure(message: string): void {
  console.log(`  ✗ [DB] ${message}`);
}
