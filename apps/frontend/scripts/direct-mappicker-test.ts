import { chromium } from '@playwright/test';

/**
 * Direct test that bypasses login/onboarding and directly tests MapPicker
 * by manipulating localStorage to simulate an authenticated state
 */
async function directMapPickerTest() {
  console.log('=== Direct MapPicker Layout Test (320x568) ===\n');

  const browser = await chromium.launch({
    headless: false,
    args: ['--disable-web-security']
  });

  const context = await browser.newContext({
    viewport: { width: 320, height: 568 },
    geolocation: { latitude: 28.6315, longitude: 77.2167 },
    permissions: ['geolocation']
  });

  const page = await context.newPage();

  let step = 1;
  const screenshot = async (name: string) => {
    const filename = `direct-${step++}-${name}.png`;
    await page.screenshot({ path: filename, fullPage: false });
    console.log(`[Screenshot] ${filename}`);
  };

  page.on('console', msg => {
    if (msg.type() === 'error' && !msg.text().includes('fonts')) {
      console.log('[Console Error]', msg.text());
    }
  });

  try {
    // Create test account
    console.log('Creating test account...');
    const email = `direct_${Date.now()}@floodsafe.test`;
    const password = 'TestPassword123!';

    const registerResponse = await fetch('http://localhost:8000/api/auth/register/email', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password, username: email.split('@')[0] })
    });

    if (!registerResponse.ok) {
      throw new Error(`Registration failed: ${registerResponse.status}`);
    }

    const { access_token, user } = await registerResponse.json();
    console.log(`  Created: ${email}`);

    // Complete profile via API (skip onboarding)
    console.log('Completing profile via API...');
    await fetch('http://localhost:8000/api/profile', {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${access_token}`
      },
      body: JSON.stringify({
        username: email.split('@')[0],
        city: 'delhi',
        notification_radius: 2.0,
        profile_complete: true
      })
    });

    // Navigate and set auth token
    console.log('\nNavigating to app...');
    await page.goto('http://localhost:5175', { waitUntil: 'domcontentloaded' });

    // Set auth token in localStorage
    await page.evaluate((data) => {
      localStorage.setItem('auth_token', data.token);
      localStorage.setItem('user', JSON.stringify(data.user));
    }, { token: access_token, user: { ...user, profile_complete: true, city: 'delhi' } });

    // Reload to apply auth state
    await page.reload({ waitUntil: 'networkidle' });
    await page.waitForTimeout(2000);
    await screenshot('home-authenticated');

    // Check if we're on home screen
    const homeVisible = await page.locator('text=Flood Risk').isVisible({ timeout: 3000 }).catch(() => false);
    console.log(`Home screen visible: ${homeVisible}`);

    // Navigate to Report screen
    console.log('\nNavigating to Report...');

    // Click the middle (Report) button in bottom nav
    const navButtons = page.locator('nav[data-bottom-nav] button');
    const buttonCount = await navButtons.count();
    console.log(`  Found ${buttonCount} nav buttons`);

    if (buttonCount >= 3) {
      // Click the 3rd button (Report FAB)
      await navButtons.nth(2).click();
      await page.waitForTimeout(1500);
    }

    await screenshot('report-screen');

    // Check what's on screen
    const pageContent = await page.content();
    const hasReportElements = pageContent.includes('Report') || pageContent.includes('flooding');
    console.log(`  Has Report elements: ${hasReportElements}`);

    // Look for location picker trigger - "Select from Map" button
    console.log('\nLooking for location picker...');

    // First, if there's a "Next" button to advance steps, click it
    for (let i = 0; i < 3; i++) {
      const nextBtn = page.locator('button:has-text("Next")').first();
      if (await nextBtn.isVisible({ timeout: 1000 }).catch(() => false)) {
        await nextBtn.click();
        await page.waitForTimeout(800);
        console.log(`  Advanced to step ${i + 2}`);
      }
    }

    // Now look for "Select from Map" button
    const mapPickerTriggers = [
      'button:has-text("Select from Map")',
      'button:has-text("Select Location")',
      'text=Select from Map',
      'button >> text=Map'
    ];

    for (const selector of mapPickerTriggers) {
      const el = page.locator(selector).first();
      if (await el.isVisible({ timeout: 1500 }).catch(() => false)) {
        console.log(`  Found: ${selector}`);
        await el.click();
        await page.waitForTimeout(2000);
        break;
      }
    }

    await screenshot('mappicker-opened');

    // Analyze MapPicker layout
    console.log('\n=== MapPicker Layout Analysis ===');

    // Check for MapPicker elements
    const mapPickerPanel = page.locator('[style*="position: fixed"][style*="z-index: 61"]');
    const panelVisible = await mapPickerPanel.isVisible({ timeout: 3000 }).catch(() => false);
    console.log(`MapPicker panel visible: ${panelVisible}`);

    if (panelVisible) {
      const panelBox = await mapPickerPanel.boundingBox();
      if (panelBox) {
        console.log(`\nPanel dimensions:`);
        console.log(`  Top: ${panelBox.y}px`);
        console.log(`  Left: ${panelBox.x}px`);
        console.log(`  Width: ${panelBox.width}px`);
        console.log(`  Height: ${panelBox.height}px`);
        console.log(`  Bottom edge: ${panelBox.y + panelBox.height}px`);
        console.log(`  Viewport: 568px`);
        console.log(`  BottomNav starts at: ${568 - 64}px = 504px`);

        const footerOverlap = (panelBox.y + panelBox.height) - 504;
        if (footerOverlap > 0) {
          console.log(`\n⚠️ OVERLAP DETECTED: Panel extends ${footerOverlap}px into BottomNav!`);
        } else {
          console.log(`\n✓ Panel fits correctly (${-footerOverlap}px gap to BottomNav)`);
        }
      }
    }

    // Check footer buttons visibility
    const confirmBtn = page.locator('button:has-text("Confirm")');
    const cancelBtn = page.locator('button:has-text("Cancel")');

    const confirmVisible = await confirmBtn.isVisible({ timeout: 2000 }).catch(() => false);
    const cancelVisible = await cancelBtn.isVisible({ timeout: 2000 }).catch(() => false);

    console.log(`\nFooter buttons:`);
    console.log(`  Confirm visible: ${confirmVisible}`);
    console.log(`  Cancel visible: ${cancelVisible}`);

    if (confirmVisible) {
      const confirmBox = await confirmBtn.boundingBox();
      if (confirmBox) {
        console.log(`  Confirm button bottom: ${confirmBox.y + confirmBox.height}px`);
        if (confirmBox.y + confirmBox.height > 504) {
          console.log(`  ⚠️ Confirm button extends into BottomNav!`);
        }
      }
    }

    await screenshot('mappicker-layout');

    // Take screenshot with page scrolled
    await page.evaluate(() => {
      const panels = document.querySelectorAll('[style*="z-index: 61"]');
      panels.forEach(p => {
        (p as HTMLElement).scrollTop = (p as HTMLElement).scrollHeight;
      });
    });
    await screenshot('mappicker-scrolled');

    console.log('\n=== Test Complete ===');
    console.log('Review direct-*.png files\n');

    // Keep browser open
    console.log('Browser open for 30 seconds for inspection...');
    await page.waitForTimeout(30000);

  } catch (error) {
    console.error('Error:', error);
    await screenshot('error');
  } finally {
    await browser.close();
  }
}

directMapPickerTest();
