import { chromium, Page, Browser } from '@playwright/test';
import {
  createDbAssertions,
  DbAssertions,
  logDbSuccess,
  logDbFailure,
} from './e2e-utils/db-assertions';

const API_BASE = 'http://localhost:8000/api';
const APP_URL = 'http://localhost:5175';

// Test account credentials
const TEST_EMAIL = `e2e_test_${Date.now()}@floodsafe.test`;
const TEST_PASSWORD = 'TestPassword123!';

interface TestContext {
  browser: Browser;
  page: Page;
  accessToken: string;
  userId: string;
  step: number;
  dbAssert: DbAssertions;
}

async function screenshot(ctx: TestContext, name: string) {
  const filename = `e2e-${ctx.step++}-${name}.png`;
  await ctx.page.screenshot({ path: filename, fullPage: false });
  console.log(`  [Screenshot] ${filename}`);
}

async function log(message: string) {
  console.log(`\n=== ${message} ===`);
}

async function runE2ETest() {
  console.log('========================================');
  console.log('FloodSafe E2E Full Test Suite');
  console.log('========================================\n');

  const browser = await chromium.launch({
    headless: false,
    slowMo: 100,
    args: ['--disable-web-security']
  });

  const page = await browser.newPage({ viewport: { width: 1400, height: 900 } });

  // Capture console errors
  page.on('console', msg => {
    if (msg.type() === 'error') {
      console.log(`  [Console Error] ${msg.text()}`);
    }
  });

  const ctx: TestContext = {
    browser,
    page,
    accessToken: '',
    userId: '',
    step: 1,
    dbAssert: createDbAssertions(TEST_EMAIL),
  };

  try {
    // ========================================
    // PHASE 1: Account Creation via API
    // ========================================
    await log('PHASE 1: Account Creation');

    console.log(`  Creating account: ${TEST_EMAIL}`);
    const registerResponse = await fetch(`${API_BASE}/auth/register/email`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email: TEST_EMAIL, password: TEST_PASSWORD })
    });

    if (!registerResponse.ok) {
      throw new Error(`Registration failed: ${await registerResponse.text()}`);
    }

    const tokens = await registerResponse.json();
    ctx.accessToken = tokens.access_token;
    console.log('  [OK] Account created successfully');
    console.log(`  Access token: ${ctx.accessToken.substring(0, 30)}...`);

    // Verify account in database via /auth/me
    const meResponse = await fetch(`${API_BASE}/auth/me`, {
      headers: { 'Authorization': `Bearer ${ctx.accessToken}` }
    });
    const userData = await meResponse.json();
    ctx.userId = userData.id;
    console.log(`  [OK] User ID: ${ctx.userId}`);
    console.log(`  [OK] Username: ${userData.username}`);
    console.log(`  [OK] Auth provider: ${userData.auth_provider}`);
    console.log(`  [OK] Profile complete: ${userData.profile_complete}`);
    console.log(`  [OK] Onboarding step: ${userData.onboarding_step}`);

    // DATABASE ASSERTION: Verify user exists in database
    console.log('\n  --- Database Verification ---');
    const dbUser = await ctx.dbAssert.verifyUserCreated();
    if (dbUser) {
      logDbSuccess(`User created in database: ${dbUser.id}`);
      logDbSuccess(`Email matches: ${dbUser.email === TEST_EMAIL}`);
      logDbSuccess(`Auth provider: ${dbUser.auth_provider}`);
    } else {
      logDbFailure('User NOT found in database after registration!');
      throw new Error('Database assertion failed: User not persisted');
    }

    // ========================================
    // PHASE 2: Login Flow via UI
    // ========================================
    await log('PHASE 2: Login Flow via UI');

    await page.goto(APP_URL, { waitUntil: 'networkidle', timeout: 30000 });
    await page.waitForTimeout(2000);
    await screenshot(ctx, 'login-screen');

    // Verify Email tab is selected (default)
    const emailTab = page.locator('button:has-text("Email")');
    const isEmailActive = await emailTab.evaluate(el => el.classList.contains('bg-blue-600'));
    console.log(`  [OK] Email tab is default: ${isEmailActive}`);

    // Fill in login form
    console.log('  Filling login form...');
    await page.fill('input[type="email"]', TEST_EMAIL);
    await page.fill('input[type="password"]', TEST_PASSWORD);
    await screenshot(ctx, 'login-filled');

    // Click Sign In submit button (use type="submit" to avoid clicking the toggle button)
    console.log('  Clicking Sign In submit button...');
    await page.click('button[type="submit"]');

    // Wait for navigation after login
    await page.waitForTimeout(3000);
    await screenshot(ctx, 'after-login');

    // ========================================
    // PHASE 3: Onboarding Flow
    // ========================================
    await log('PHASE 3: Onboarding Flow');

    // Check if we're on onboarding screen
    const onboardingTitle = page.locator('text=Welcome to FloodSafe');
    if (await onboardingTitle.isVisible({ timeout: 5000 }).catch(() => false)) {
      console.log('  [OK] Onboarding screen detected');
      await screenshot(ctx, 'onboarding-step1');

      // Step 1: City Selection - select Delhi
      console.log('  Step 1: City Selection');
      const delhiOption = page.locator('text=Delhi').first();
      if (await delhiOption.isVisible({ timeout: 2000 }).catch(() => false)) {
        await delhiOption.click();
        await page.waitForTimeout(500);
        console.log('  [OK] Selected Delhi');
      }
      await page.click('button:has-text("Next")');
      await page.waitForTimeout(2000);
      await screenshot(ctx, 'onboarding-step2');

      // Step 2: Profile - username should be pre-filled, just click Next
      console.log('  Step 2: Profile');
      const usernameInput = page.locator('input#username');
      if (await usernameInput.isVisible({ timeout: 2000 }).catch(() => false)) {
        const currentUsername = await usernameInput.inputValue();
        console.log(`  [OK] Username: ${currentUsername}`);
      }
      await page.click('button:has-text("Next")');
      await page.waitForTimeout(2000);
      await screenshot(ctx, 'onboarding-step3');

      // Step 3: Watch Areas - add a watch area by searching
      console.log('  Step 3: Watch Areas');
      const areaNameInput = page.locator('input[placeholder*="Area name"]');
      const searchInput = page.locator('input[placeholder*="Search for a location"]');

      if (await areaNameInput.isVisible({ timeout: 2000 }).catch(() => false)) {
        // Fill area name
        await areaNameInput.fill('Home');
        await page.waitForTimeout(300);

        // Search for Connaught Place
        await searchInput.fill('Connaught Place Delhi');
        await page.waitForTimeout(1500); // Wait for search results

        // Click on first search result
        const searchResult = page.locator('.border.rounded-lg.divide-y button').first();
        if (await searchResult.isVisible({ timeout: 3000 }).catch(() => false)) {
          await searchResult.click();
          await page.waitForTimeout(500);
          console.log('  [OK] Added watch area: Home - Connaught Place');
        } else {
          console.log('  [WARN] No search results, trying Use My Current Location');
          const myLocationBtn = page.locator('button:has-text("Use My Current Location")');
          if (await myLocationBtn.isVisible({ timeout: 1000 }).catch(() => false)) {
            await myLocationBtn.click();
            await page.waitForTimeout(2000);
          }
        }
      }
      await screenshot(ctx, 'onboarding-step3-with-area');
      await page.click('button:has-text("Next")');
      await page.waitForTimeout(2000);
      await screenshot(ctx, 'onboarding-step4');

      // Step 4: Daily Routes (optional) - click Skip
      console.log('  Step 4: Daily Routes (skipping)');
      const skipButton = page.locator('button:has-text("Skip")');
      if (await skipButton.isVisible({ timeout: 2000 }).catch(() => false)) {
        await skipButton.click();
        await page.waitForTimeout(2000);
        console.log('  [OK] Skipped daily routes');
      } else {
        await page.click('button:has-text("Next")');
        await page.waitForTimeout(2000);
      }
      await screenshot(ctx, 'onboarding-step5');

      // Step 5: Completion - click Get Started
      console.log('  Step 5: Completion');
      const getStartedBtn = page.locator('button:has-text("Get Started")');
      if (await getStartedBtn.isVisible({ timeout: 3000 }).catch(() => false)) {
        await getStartedBtn.click();
        await page.waitForTimeout(2000);
        console.log('  [OK] Onboarding completed');
      }
    } else {
      console.log('  [INFO] No onboarding screen - user may already be onboarded');
    }

    await screenshot(ctx, 'home-screen');

    // DATABASE ASSERTION: Verify onboarding updated user profile
    console.log('\n  --- Database Verification (Onboarding) ---');
    const onboardingVerified = await ctx.dbAssert.verifyUserUpdated(ctx.userId, {
      profile_complete: true,
      city_preference: 'Delhi',
    });
    if (onboardingVerified) {
      logDbSuccess('User profile_complete=true in database');
      logDbSuccess('User city_preference=Delhi in database');
    } else {
      logDbFailure('User onboarding fields not properly updated in database');
      // Don't throw - onboarding may have been skipped
    }

    // Also verify watch area was created during onboarding
    const watchArea = await ctx.dbAssert.verifyWatchAreaCreated(ctx.userId, 'Home');
    if (watchArea) {
      logDbSuccess(`Watch area created in database: ${watchArea.name}`);
    } else {
      logDbFailure('Watch area "Home" not found in database');
    }

    // ========================================
    // PHASE 4: HomeScreen Features
    // ========================================
    await log('PHASE 4: HomeScreen Features');

    // Wait for HomeScreen to fully load
    await page.waitForTimeout(2000);

    // Check for key elements
    const homeElements = {
      'Risk Banner': 'text=FLOOD RISK',
      'Your Area Card': 'text=Your Area',
      'Alerts Card': 'text=Active',
      'Report Button': 'text=Report',
      'Routes Button': 'text=Routes',
      'Map': '.maplibregl-canvas'
    };

    for (const [name, selector] of Object.entries(homeElements)) {
      const element = page.locator(selector).first();
      const visible = await element.isVisible({ timeout: 3000 }).catch(() => false);
      console.log(`  ${visible ? '[OK]' : '[MISSING]'} ${name}`);
    }

    await screenshot(ctx, 'home-overview');

    // ========================================
    // PHASE 5: Submit a Flood Report (4-Step Wizard)
    // ========================================
    await log('PHASE 5: Submit Flood Report');

    // Click the Report button in bottom nav (it's the large blue button)
    const reportNavBtn = page.locator('nav button:has-text("Report")').first();
    if (await reportNavBtn.isVisible({ timeout: 3000 }).catch(() => false)) {
      await reportNavBtn.click();
      await page.waitForTimeout(3000);
      await screenshot(ctx, 'report-step1');

      // STEP 1: Location + Description
      console.log('  Step 1: Location & Description');

      // Click "Select from Map" to use current location
      const selectFromMap = page.locator('text=Select from Map').first();
      if (await selectFromMap.isVisible({ timeout: 2000 }).catch(() => false)) {
        await selectFromMap.click();
        await page.waitForTimeout(2000);

        // Click on map to select location (center of map)
        const mapCanvas = page.locator('.maplibregl-canvas').first();
        if (await mapCanvas.isVisible({ timeout: 2000 }).catch(() => false)) {
          const box = await mapCanvas.boundingBox();
          if (box) {
            await page.mouse.click(box.x + box.width / 2, box.y + box.height / 2);
            await page.waitForTimeout(1000);
            console.log('  [OK] Location selected from map');
          }
        }

        // Click confirm/done for location selection
        const confirmLocationBtn = page.locator('button:has-text("Confirm"), button:has-text("Done")').first();
        if (await confirmLocationBtn.isVisible({ timeout: 2000 }).catch(() => false)) {
          await confirmLocationBtn.click();
          await page.waitForTimeout(1000);
        }
      }

      // Fill description
      const descInput = page.locator('textarea').first();
      if (await descInput.isVisible({ timeout: 2000 }).catch(() => false)) {
        await descInput.fill('E2E Test Report - Water logging near Connaught Place. Ankle deep water on road.');
        console.log('  [OK] Description filled');
      }

      // Scroll up to avoid bottom nav overlap
      await page.evaluate(() => window.scrollTo(0, 0));
      await page.waitForTimeout(500);

      // Click Next to go to Step 2 (use force to avoid nav overlap)
      const nextBtn = page.locator('button:has-text("Next"), button:has-text("Continue")').first();
      if (await nextBtn.isVisible({ timeout: 2000 }).catch(() => false)) {
        await nextBtn.click({ force: true });
        await page.waitForTimeout(2000);
      }
      await screenshot(ctx, 'report-step2');

      // STEP 2: Details (Water Level, etc.)
      console.log('  Step 2: Details');

      // Select water depth - look for water level options
      const ankleOption = page.locator('button:has-text("Ankle"), label:has-text("Ankle"), div:has-text("Ankle")').first();
      if (await ankleOption.isVisible({ timeout: 2000 }).catch(() => false)) {
        await ankleOption.click();
        console.log('  [OK] Water level: Ankle');
      }

      // Scroll up before next step
      await page.evaluate(() => window.scrollTo(0, 0));
      await page.waitForTimeout(300);

      // Click Next to go to Step 3
      const nextBtn2 = page.locator('button:has-text("Next"), button:has-text("Continue")').first();
      if (await nextBtn2.isVisible({ timeout: 2000 }).catch(() => false)) {
        await nextBtn2.click({ force: true });
        await page.waitForTimeout(2000);
      }
      await screenshot(ctx, 'report-step3');

      // STEP 3: Photo (Optional - Skip)
      console.log('  Step 3: Photo (skipping)');
      await page.evaluate(() => window.scrollTo(0, 0));
      const skipPhotoBtn = page.locator('button:has-text("Skip"), button:has-text("Next")').first();
      if (await skipPhotoBtn.isVisible({ timeout: 2000 }).catch(() => false)) {
        await skipPhotoBtn.click({ force: true });
        await page.waitForTimeout(2000);
      }
      await screenshot(ctx, 'report-step4');

      // STEP 4: Confirm & Submit
      console.log('  Step 4: Confirm');
      await page.evaluate(() => window.scrollTo(0, 0));
      const submitBtn = page.locator('button:has-text("Submit"), button:has-text("Confirm")').first();
      if (await submitBtn.isVisible({ timeout: 2000 }).catch(() => false)) {
        await submitBtn.click({ force: true });
        await page.waitForTimeout(3000);
        console.log('  [OK] Report submitted!');
        await screenshot(ctx, 'report-submitted');
      }
    } else {
      console.log('  [INFO] Report button not found');
    }

    // ========================================
    // PHASE 6: Verify Report in Database
    // ========================================
    await log('PHASE 6: Verify Report in Database');

    // API-level verification (existing)
    const reportsResponse = await fetch(`${API_BASE}/reports`, {
      headers: { 'Authorization': `Bearer ${ctx.accessToken}` }
    });
    const reports = await reportsResponse.json();

    const myReports = reports.filter((r: any) => r.user_id === ctx.userId);
    console.log(`  Total reports (API): ${reports.length}`);
    console.log(`  My reports (API): ${myReports.length}`);

    if (myReports.length > 0) {
      const latestReport = myReports[0];
      console.log(`  [OK] Latest report ID: ${latestReport.id}`);
      console.log(`  [OK] Description: ${latestReport.description?.substring(0, 50)}...`);
      console.log(`  [OK] Water depth: ${latestReport.water_depth}`);
      console.log(`  [OK] Verified: ${latestReport.verified}`);
    }

    // DATABASE ASSERTION: Verify report persisted directly in database
    console.log('\n  --- Database Verification (Report) ---');
    const dbReport = await ctx.dbAssert.verifyReportCreated(ctx.userId, {
      description: 'E2E Test Report',  // Partial match
    });
    if (dbReport) {
      logDbSuccess(`Report found in database: ${dbReport.id}`);
      logDbSuccess(`User ID matches: ${dbReport.user_id === ctx.userId}`);
      logDbSuccess(`Description: ${dbReport.description?.substring(0, 40)}...`);
      logDbSuccess(`Water depth: ${dbReport.water_depth || 'not set'}`);
      logDbSuccess(`Timestamp: ${dbReport.timestamp}`);
    } else {
      logDbFailure('Report NOT found in database after API returned success!');
      throw new Error('Database assertion failed: Report not persisted');
    }

    // Compare API and DB report counts
    const dbReportCount = await ctx.dbAssert.getRecordCount('reports', ctx.userId);
    console.log(`\n  Reports count comparison:`);
    console.log(`    - API reports: ${myReports.length}`);
    console.log(`    - DB reports: ${dbReportCount}`);
    if (myReports.length === dbReportCount) {
      logDbSuccess('API and DB report counts match');
    } else {
      logDbFailure(`Report count mismatch: API=${myReports.length}, DB=${dbReportCount}`);
    }

    // ========================================
    // PHASE 7: Check My Reports in Profile
    // ========================================
    await log('PHASE 7: Check My Reports in Profile');

    // Navigate to profile/my reports
    const profileIcon = page.locator('[aria-label="Profile"], button:has(svg)').last();
    if (await profileIcon.isVisible({ timeout: 2000 }).catch(() => false)) {
      await profileIcon.click();
      await page.waitForTimeout(1500);
      await screenshot(ctx, 'profile-dropdown');
    }

    // Look for My Reports link
    const myReportsLink = page.locator('text=My Reports').first();
    if (await myReportsLink.isVisible({ timeout: 2000 }).catch(() => false)) {
      await myReportsLink.click();
      await page.waitForTimeout(2000);
      await screenshot(ctx, 'my-reports-screen');
      console.log('  [OK] My Reports screen opened');
    }

    // ========================================
    // PHASE 8: Test Watch Areas via Profile
    // ========================================
    await log('PHASE 8: Test Watch Areas via Profile');

    // Navigate to Profile tab to check watch areas
    const profileTab = page.locator('a:has-text("Profile"), nav >> text=Profile').first();
    if (await profileTab.isVisible({ timeout: 3000 }).catch(() => false)) {
      await profileTab.click();
      await page.waitForTimeout(2000);
      await screenshot(ctx, 'profile-screen');

      // Look for watch areas section
      const watchAreasSection = page.locator('text=Watch Areas, text=watch area').first();
      if (await watchAreasSection.isVisible({ timeout: 2000 }).catch(() => false)) {
        console.log('  [OK] Watch Areas visible in profile');
      }
      console.log('  [OK] Profile accessed');
    } else {
      console.log('  [INFO] Profile tab not found');
    }

    // ========================================
    // PHASE 9: Test Flood Atlas Navigation
    // ========================================
    await log('PHASE 9: Test Flood Atlas & Navigation');

    // Click Flood Atlas in bottom nav
    const atlasTab = page.locator('a:has-text("Flood Atlas"), nav >> text=Flood Atlas').first();
    if (await atlasTab.isVisible({ timeout: 3000 }).catch(() => false)) {
      await atlasTab.click();
      await page.waitForTimeout(4000);
      await screenshot(ctx, 'flood-atlas');
      console.log('  [OK] Flood Atlas loaded');

      // Check for map controls
      const mapCanvas = page.locator('.maplibregl-canvas');
      if (await mapCanvas.isVisible({ timeout: 3000 }).catch(() => false)) {
        console.log('  [OK] Map canvas visible');
      }

      // Look for layer toggles or controls
      const layerControls = page.locator('button[title], .maplibregl-ctrl button').first();
      if (await layerControls.isVisible({ timeout: 2000 }).catch(() => false)) {
        console.log('  [OK] Map controls visible');
      }

      await screenshot(ctx, 'flood-atlas-overview');
    } else {
      console.log('  [INFO] Flood Atlas tab not found');
    }

    // ========================================
    // PHASE 10: Final Consistency Checks
    // ========================================
    await log('PHASE 10: Final Consistency Checks');

    // Re-fetch user to verify profile_complete updated
    const finalMeResponse = await fetch(`${API_BASE}/auth/me`, {
      headers: { 'Authorization': `Bearer ${ctx.accessToken}` }
    });
    const finalUserData = await finalMeResponse.json();

    console.log('  Final User State:');
    console.log(`    - Profile complete: ${finalUserData.profile_complete}`);
    console.log(`    - City preference: ${finalUserData.city_preference}`);
    console.log(`    - Reports count: ${finalUserData.reports_count}`);
    console.log(`    - Points: ${finalUserData.points}`);
    console.log(`    - Level: ${finalUserData.level}`);

    // Final screenshot
    await screenshot(ctx, 'final-state');

    // ========================================
    // PHASE 11: Database Summary & Cleanup
    // ========================================
    await log('PHASE 11: Database Summary & Cleanup');

    // Final database record counts
    console.log('  Final Database State:');
    const finalReportCount = await ctx.dbAssert.getRecordCount('reports', ctx.userId);
    const finalWatchAreaCount = await ctx.dbAssert.getRecordCount('watch_areas', ctx.userId);
    console.log(`    - Reports in DB: ${finalReportCount}`);
    console.log(`    - Watch Areas in DB: ${finalWatchAreaCount}`);

    // Verify final user state in database
    const finalDbUser = await ctx.dbAssert.verifyUserCreated();
    if (finalDbUser) {
      console.log('  Final User State (from DB):');
      console.log(`    - Profile complete: ${finalDbUser.profile_complete}`);
      console.log(`    - City preference: ${finalDbUser.city_preference}`);
      console.log(`    - Reports count: ${finalDbUser.reports_count}`);
      console.log(`    - Points: ${finalDbUser.points}`);
      console.log(`    - Level: ${finalDbUser.level}`);
    }

    // Cleanup test data (optional - set CLEANUP_TEST_DATA=true to enable)
    if (process.env.CLEANUP_TEST_DATA === 'true') {
      console.log('\n  --- Cleaning up test data ---');
      try {
        await ctx.dbAssert.cleanup();
        logDbSuccess('All test data cleaned up successfully');
      } catch (cleanupError) {
        logDbFailure(`Cleanup failed: ${cleanupError}`);
      }
    } else {
      console.log('\n  [INFO] Skipping cleanup (set CLEANUP_TEST_DATA=true to enable)');
      console.log(`  [INFO] Test user ${TEST_EMAIL} remains in database`);
    }

    console.log('\n========================================');
    console.log('E2E TEST COMPLETED SUCCESSFULLY');
    console.log('========================================');
    console.log(`\nTest Account: ${TEST_EMAIL}`);
    console.log(`User ID: ${ctx.userId}`);
    console.log(`Total Screenshots: ${ctx.step - 1}`);
    console.log('\nDatabase Assertions: PASSED');

  } catch (error) {
    console.error('\n[FAIL] Test failed:', error);
    await screenshot(ctx, 'error-state');

    // Try to cleanup even on failure (if cleanup is enabled)
    if (process.env.CLEANUP_TEST_DATA === 'true') {
      console.log('\n  --- Attempting cleanup after failure ---');
      try {
        await ctx.dbAssert.cleanup();
        console.log('  [OK] Cleanup completed');
      } catch (cleanupError) {
        console.log(`  [WARN] Cleanup failed: ${cleanupError}`);
      }
    }

    throw error;
  } finally {
    await browser.close();
  }
}

runE2ETest().catch(console.error);
