import { chromium } from '@playwright/test';

async function mobileReportTest() {
  console.log('Launching browser at mobile viewport (320x568)...');
  const browser = await chromium.launch({
    headless: false,
    args: ['--disable-web-security']
  });

  // Mobile viewport - iPhone SE size
  const page = await browser.newPage({ viewport: { width: 320, height: 568 } });

  page.on('console', msg => {
    if (msg.type() === 'error') console.log('[Console Error]', msg.text());
  });

  let step = 1;
  const screenshot = async (name: string) => {
    const filename = `mobile-${step++}-${name}.png`;
    await page.screenshot({ path: filename, fullPage: false });
    console.log(`[Screenshot] ${filename}`);
  };

  try {
    // Step 1: Create test account via API
    console.log('\n=== Creating Test Account ===');
    const email = `mobile_test_${Date.now()}@floodsafe.test`;
    const password = 'TestPassword123!';

    const registerResponse = await fetch('http://localhost:8000/api/auth/register/email', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password, username: email.split('@')[0] })
    });

    if (!registerResponse.ok) {
      throw new Error(`Failed to create account: ${registerResponse.status}`);
    }

    const { access_token: _token } = await registerResponse.json();
    console.log(`  [OK] Account created: ${email}`);

    // Step 2: Navigate and login
    console.log('\n=== Login Flow ===');
    await page.goto('http://localhost:5175', { waitUntil: 'networkidle', timeout: 30000 });
    await page.waitForTimeout(1000);
    await screenshot('login-screen');

    // Fill login form
    await page.fill('input[type="email"]', email);
    await page.fill('input[type="password"]', password);
    await screenshot('login-filled');

    await page.click('button:has-text("Sign In")');
    await page.waitForTimeout(2000);
    await screenshot('after-login');

    // Step 3: Complete onboarding quickly
    console.log('\n=== Quick Onboarding ===');

    // Step 1: City
    if (await page.locator('text=Select your city').isVisible({ timeout: 3000 }).catch(() => false)) {
      await page.click('text=Delhi');
      await page.click('button:has-text("Next")');
      await page.waitForTimeout(500);
    }

    // Step 2: Profile - just click next
    if (await page.locator('button:has-text("Next")').isVisible({ timeout: 2000 }).catch(() => false)) {
      await page.click('button:has-text("Next")');
      await page.waitForTimeout(500);
    }

    // Step 3: Watch Areas - skip
    if (await page.locator('button:has-text("Skip"), button:has-text("Next")').first().isVisible({ timeout: 2000 }).catch(() => false)) {
      await page.click('button:has-text("Skip"), button:has-text("Next")');
      await page.waitForTimeout(500);
    }

    // Step 4: Routes - skip
    if (await page.locator('button:has-text("Skip"), button:has-text("Next")').first().isVisible({ timeout: 2000 }).catch(() => false)) {
      await page.click('button:has-text("Skip"), button:has-text("Next")');
      await page.waitForTimeout(500);
    }

    // Step 5: Complete
    if (await page.locator('button:has-text("Get Started"), button:has-text("Complete")').first().isVisible({ timeout: 2000 }).catch(() => false)) {
      await page.click('button:has-text("Get Started"), button:has-text("Complete")');
      await page.waitForTimeout(1000);
    }

    await screenshot('home-screen');
    console.log('  [OK] Onboarding complete');

    // Step 4: Click Report button (FAB)
    console.log('\n=== Report Flow ===');
    const reportBtn = page.locator('button:has-text("Report"), [data-bottom-nav] button:nth-child(3)').first();
    if (await reportBtn.isVisible({ timeout: 3000 }).catch(() => false)) {
      await reportBtn.click();
      await page.waitForTimeout(1000);
      await screenshot('report-step1');
      console.log('  [OK] Report screen opened');
    }

    // Step 5: Click on Location field to open MapPicker
    console.log('\n=== Opening MapPicker ===');
    const _locationField = page.locator('text=Select location, button:has-text("Location"), [class*="location"]').first();

    // Try clicking on any location-related element
    const locationTriggers = [
      'button:has-text("Select location")',
      'button:has-text("Location")',
      'text=Select location',
      'text=Click to select',
      '.cursor-pointer:has-text("location")'
    ];

    for (const selector of locationTriggers) {
      try {
        if (await page.locator(selector).first().isVisible({ timeout: 1000 }).catch(() => false)) {
          console.log(`  Found: ${selector}`);
          await page.locator(selector).first().click();
          await page.waitForTimeout(1500);
          break;
        }
      } catch (e) {
        continue;
      }
    }

    await screenshot('mappicker-open');
    console.log('  [OK] MapPicker should be open');

    // Step 6: Take multiple screenshots to see the layout
    await page.waitForTimeout(2000);
    await screenshot('mappicker-loaded');

    // Scroll to see footer
    await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
    await screenshot('mappicker-scrolled');

    console.log('\n=== Test Complete ===');
    console.log('Check the mobile-*.png files for the MapPicker layout issue');

    // Keep browser open for manual inspection
    console.log('\nBrowser will stay open for 30 seconds for manual inspection...');
    await page.waitForTimeout(30000);

  } catch (error) {
    console.error('Error:', error);
    await screenshot('error');
  } finally {
    await browser.close();
  }
}

mobileReportTest();
