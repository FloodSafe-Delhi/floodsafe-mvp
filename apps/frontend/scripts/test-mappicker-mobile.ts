import { chromium } from '@playwright/test';

async function testMapPickerMobile() {
  console.log('Launching browser at mobile viewport (320x568)...');
  const browser = await chromium.launch({
    headless: false,
    args: ['--disable-web-security']
  });

  // Mobile viewport - iPhone SE size
  const context = await browser.newContext({
    viewport: { width: 320, height: 568 },
    // Mock geolocation to Delhi (Connaught Place)
    geolocation: { latitude: 28.6315, longitude: 77.2167 },
    permissions: ['geolocation']
  });
  const page = await context.newPage();

  page.on('console', msg => {
    if (msg.type() === 'error') console.log('[Console Error]', msg.text());
  });

  let step = 1;
  const screenshot = async (name: string) => {
    const filename = `mappicker-mobile-${step++}-${name}.png`;
    await page.screenshot({ path: filename, fullPage: false });
    console.log(`[Screenshot] ${filename}`);
  };

  try {
    // Step 1: Create test account via API
    console.log('\n=== Creating Test Account ===');
    const email = `mp_test_${Date.now()}@floodsafe.test`;
    const password = 'TestPassword123!';

    const registerResponse = await fetch('http://localhost:8000/api/auth/register/email', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password, username: email.split('@')[0] })
    });

    if (!registerResponse.ok) {
      const errorText = await registerResponse.text();
      throw new Error(`Failed to create account: ${registerResponse.status} - ${errorText}`);
    }

    const { access_token: _token } = await registerResponse.json();
    console.log(`  [OK] Account created: ${email}`);

    // Step 2: Navigate to app
    console.log('\n=== Navigating to App ===');
    await page.goto('http://localhost:5175', { waitUntil: 'networkidle', timeout: 30000 });
    await page.waitForTimeout(2000);
    await screenshot('initial-load');

    // Step 3: Fill and submit login form
    console.log('\n=== Login Flow ===');
    await page.waitForSelector('input[type="email"]', { timeout: 10000 });
    await page.fill('input[type="email"]', email);
    await page.fill('input[type="password"]', password);
    await screenshot('login-filled');

    const signInBtn = page.locator('button[type="submit"]:has-text("Sign In")');
    await signInBtn.click();
    console.log('  [OK] Clicked Sign In');
    await page.waitForTimeout(3000);
    await screenshot('after-login');

    // Step 4: Complete full onboarding
    console.log('\n=== Completing Onboarding ===');

    // Step 1: City Selection - Click Delhi
    if (await page.locator('text=Select your city').isVisible({ timeout: 3000 }).catch(() => false)) {
      console.log('  Step 1: Selecting Delhi...');
      await page.locator('text=Delhi').first().click();
      await page.waitForTimeout(500);
      await page.locator('button:has-text("Next")').click();
      await page.waitForTimeout(1000);
    }

    // Step 2: Your Profile - Just click Next
    if (await page.locator('text=Your Profile').isVisible({ timeout: 2000 }).catch(() => false)) {
      console.log('  Step 2: Profile - clicking Next...');
      await page.locator('button:has-text("Next")').click();
      await page.waitForTimeout(1000);
    }

    // Step 3: Watch Areas - Add Connaught Place
    if (await page.locator('text=Watch Areas').isVisible({ timeout: 2000 }).catch(() => false)) {
      console.log('  Step 3: Adding Watch Area...');
      await screenshot('onboarding-watch-areas');

      // Fill area name
      const areaInput = page.locator('input[placeholder*="Area name"]');
      if (await areaInput.isVisible({ timeout: 2000 }).catch(() => false)) {
        await areaInput.fill('My Area');
      }

      // Search for Connaught Place
      const searchInput = page.locator('input[placeholder*="Search for a location"]');
      if (await searchInput.isVisible({ timeout: 2000 }).catch(() => false)) {
        await searchInput.fill('Connaught Place, Delhi');
        await page.waitForTimeout(1500);

        // Click first search result
        const firstResult = page.locator('.cursor-pointer:has-text("Connaught"), li:has-text("Connaught")').first();
        if (await firstResult.isVisible({ timeout: 3000 }).catch(() => false)) {
          await firstResult.click();
          await page.waitForTimeout(1000);
        }
      }

      // Try "Use My Current Location" button as fallback
      const useLocationBtn = page.locator('button:has-text("Use My Current Location")');
      if (await useLocationBtn.isVisible({ timeout: 2000 }).catch(() => false)) {
        await useLocationBtn.click();
        await page.waitForTimeout(2000);
      }

      await screenshot('watch-area-added');

      // Click Next/Skip
      const nextBtn = page.locator('button:has-text("Next"), button:has-text("Skip")').first();
      if (await nextBtn.isVisible({ timeout: 2000 }).catch(() => false)) {
        await nextBtn.click();
        await page.waitForTimeout(1000);
      }
    }

    // Step 4: Daily Routes - Skip
    if (await page.locator('text=Daily Routes').isVisible({ timeout: 2000 }).catch(() => false)) {
      console.log('  Step 4: Daily Routes - skipping...');
      const skipBtn = page.locator('button:has-text("Skip"), button:has-text("Next")').first();
      await skipBtn.click();
      await page.waitForTimeout(1000);
    }

    // Step 5: Complete
    const completeBtn = page.locator('button:has-text("Get Started"), button:has-text("Complete")').first();
    if (await completeBtn.isVisible({ timeout: 2000 }).catch(() => false)) {
      console.log('  Step 5: Completing onboarding...');
      await completeBtn.click();
      await page.waitForTimeout(2000);
    }

    await screenshot('home-screen');
    console.log('  [OK] Onboarding complete');

    // Step 5: Navigate to Report tab via FAB or nav
    console.log('\n=== Opening Report Screen ===');

    // Look for the Report FAB button (centered blue circle in bottom nav)
    const fabBtn = page.locator('button:has(.bg-blue-600.rounded-full), nav button >> nth=2').first();
    if (await fabBtn.isVisible({ timeout: 3000 }).catch(() => false)) {
      await fabBtn.click();
      console.log('  [OK] Clicked Report FAB');
    } else {
      // Try clicking middle item in bottom nav
      const middleNav = page.locator('nav button').nth(2);
      if (await middleNav.isVisible({ timeout: 2000 }).catch(() => false)) {
        await middleNav.click();
        console.log('  [OK] Clicked middle nav button');
      }
    }

    await page.waitForTimeout(2000);
    await screenshot('report-screen');

    // Step 6: Look for location picker in Report flow
    console.log('\n=== Finding Location Picker in Report ===');

    // The Report screen has a multi-step wizard
    // Step 1 should ask for flood type/severity
    // Step 2 asks for location

    // Check if we're on Report screen
    const hasReportTitle = await page.locator('text=Report Flooding').isVisible({ timeout: 3000 }).catch(() => false);
    console.log(`  On Report screen: ${hasReportTitle}`);

    // Advance to location step if needed
    const nextStepBtn = page.locator('button:has-text("Next")');
    if (await nextStepBtn.isVisible({ timeout: 2000 }).catch(() => false)) {
      await nextStepBtn.click();
      await page.waitForTimeout(1000);
      console.log('  [OK] Advanced to next step');
    }

    await screenshot('report-step');

    // Look for location picker trigger
    const locationTriggers = [
      'text=Select a location',
      'text=Tap to select location',
      'button:has-text("Select Location")',
      '[data-testid="location-select"]',
      'div:has-text("Location") >> button',
      '.border:has-text("location")'
    ];

    for (const selector of locationTriggers) {
      try {
        const el = page.locator(selector).first();
        if (await el.isVisible({ timeout: 1500 }).catch(() => false)) {
          console.log(`  Found: ${selector}`);
          await el.click();
          await page.waitForTimeout(2000);
          break;
        }
      } catch (e) {
        continue;
      }
    }

    await screenshot('location-picker-opened');

    // Step 7: Take screenshots of MapPicker panel
    console.log('\n=== MapPicker Layout Screenshots ===');
    await page.waitForTimeout(2000);
    await screenshot('mappicker-panel');

    // Check if MapPicker footer is visible
    const footer = page.locator('text=Selected Location');
    const footerVisible = await footer.isVisible({ timeout: 2000 }).catch(() => false);
    console.log(`  MapPicker footer visible: ${footerVisible}`);

    // Confirm and Cancel buttons
    const confirmBtn = page.locator('button:has-text("Confirm")');
    const cancelBtn = page.locator('button:has-text("Cancel")');
    const confirmVisible = await confirmBtn.isVisible({ timeout: 2000 }).catch(() => false);
    const cancelVisible = await cancelBtn.isVisible({ timeout: 2000 }).catch(() => false);
    console.log(`  Confirm button visible: ${confirmVisible}`);
    console.log(`  Cancel button visible: ${cancelVisible}`);

    await screenshot('mappicker-final');

    console.log('\n=== Test Complete ===');
    console.log('Review the mappicker-mobile-*.png files');

    // Keep browser open for inspection
    console.log('\nBrowser open for 45 seconds for manual inspection...');
    await page.waitForTimeout(45000);

  } catch (error) {
    console.error('Error:', error);
    await screenshot('error-state');
  } finally {
    await browser.close();
  }
}

testMapPickerMobile();
