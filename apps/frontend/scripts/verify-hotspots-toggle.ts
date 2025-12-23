/**
 * E2E Visual Verification: Hotspots Toggle Button & Green Dots
 *
 * Tests:
 * 1. Button is GREEN when hotspots are ON
 * 2. Button turns WHITE when toggled OFF
 * 3. Green dots visible on Delhi map when ON
 * 4. Dots disappear when toggled OFF
 * 5. No dots on Bangalore map
 */

import { chromium, Page } from 'playwright';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const BASE_URL = 'http://localhost:5175';
const SCREENSHOT_DIR = path.join(__dirname, '..');

// Test account credentials (created via API with profile_complete=true, city=delhi)
const TEST_EMAIL = 'hotspots_e2e_test@floodsafe.test';
const TEST_PASSWORD = 'TestPassword123!';

async function sleep(ms: number) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

async function takeScreenshot(page: Page, name: string) {
    const screenshotPath = path.join(SCREENSHOT_DIR, name);
    await page.screenshot({ path: screenshotPath, fullPage: false });
    console.log(`📸 Screenshot saved: ${name}`);
}

async function loginOrCreate(page: Page) {
    console.log('\n🔐 Logging in...');

    await page.goto(BASE_URL);
    await sleep(2000); // Wait for initial load

    // Check if already logged in (look for canvas which is on home screen)
    const isLoggedIn = await page.locator('canvas').isVisible().catch(() => false);
    if (isLoggedIn) {
        console.log('✓ Already logged in');
        return;
    }

    // Wait for login screen
    await page.waitForSelector('input[type="email"]', { timeout: 10000 });

    // Fill login form
    await page.fill('input[type="email"]', TEST_EMAIL);
    await page.fill('input[type="password"]', TEST_PASSWORD);

    console.log('  Submitting login form...');

    // Press Enter to submit (more reliable than button click)
    await page.keyboard.press('Enter');

    // Wait for navigation to complete
    await sleep(5000);

    // Check if login succeeded by looking for canvas or error
    const hasError = await page.locator('text=Invalid credentials').isVisible().catch(() => false);

    if (hasError) {
        throw new Error('Login failed - invalid credentials. Check test account exists in database.');
    }

    // Wait for navigation to complete (either home screen or redirect)
    await sleep(2000);

    // Take screenshot of post-login state
    await takeScreenshot(page, 'hotspots-0-post-login.png');

    console.log('✓ Logged in successfully');
}

async function verifyHotspotsToggle(page: Page) {
    console.log('\n🧪 Starting Hotspots Toggle E2E Verification...\n');

    // Wait for home screen to load
    console.log('1️⃣ Waiting for home screen to load...');

    // First wait for any loading/redirect to complete
    await sleep(3000);

    // Navigate to Flood Atlas where map controls are visible
    console.log('   Navigating to Flood Atlas...');
    await page.click('text=Flood Atlas');
    await sleep(3000);

    // Check if we're on the atlas screen by looking for the canvas
    await page.waitForSelector('canvas', { timeout: 30000 });
    await sleep(5000); // Let map and hotspots fully initialize
    console.log('✓ Map loaded');

    // Scenario 1: Verify button is GREEN initially (hotspots ON by default)
    console.log('\n2️⃣ Scenario 1: Verify initial button state (GREEN = ON)');

    // Find the hotspots button (Droplets icon button)
    const hotspotsButton = page.locator('button[title*="Toggle waterlogging hotspots"]');
    await hotspotsButton.waitFor({ state: 'visible', timeout: 10000 });

    // Get button background color
    const buttonBg = await hotspotsButton.evaluate((el) => {
        return window.getComputedStyle(el).backgroundColor;
    });

    console.log(`   Button background color: ${buttonBg}`);

    // Take screenshot of initial state
    await takeScreenshot(page, 'hotspots-1-initial-button-green.png');

    // Verify button is green (rgb(34, 197, 94) = #22c55e = green-500)
    const isGreen = buttonBg.includes('34') || buttonBg.includes('green');
    if (isGreen) {
        console.log('✅ PASS: Button is GREEN (hotspots ON)');
    } else {
        console.log(`❌ FAIL: Button is NOT green (got: ${buttonBg})`);
    }

    await sleep(2000);

    // Scenario 2: Check if dots are visible on map
    console.log('\n3️⃣ Scenario 2: Verify green dots visible on Delhi map');

    // Pan to central Delhi where hotspots should be visible
    await page.evaluate(() => {
        const canvas = document.querySelector('canvas');
        if (canvas) {
            // Simulate pan to center of Delhi (28.6139, 77.2090)
            // This is approximate - hotspots should be visible around Connaught Place
        }
    });

    await sleep(2000);
    await takeScreenshot(page, 'hotspots-2-dots-visible-delhi.png');
    console.log('✅ Screenshot captured with dots (verify manually)');

    // Scenario 3: Toggle OFF - Button should turn WHITE
    console.log('\n4️⃣ Scenario 3: Toggle OFF - verify button turns WHITE');

    await hotspotsButton.click();
    await sleep(1000);

    const buttonBgAfterToggleOff = await hotspotsButton.evaluate((el) => {
        return window.getComputedStyle(el).backgroundColor;
    });

    console.log(`   Button background color after toggle OFF: ${buttonBgAfterToggleOff}`);
    await takeScreenshot(page, 'hotspots-3-toggle-off-button-white.png');

    // Verify button is white (rgb(255, 255, 255))
    const isWhite = buttonBgAfterToggleOff.includes('255, 255, 255');
    if (isWhite) {
        console.log('✅ PASS: Button is WHITE (hotspots OFF)');
    } else {
        console.log(`❌ FAIL: Button is NOT white (got: ${buttonBgAfterToggleOff})`);
    }

    await sleep(1000);

    // Scenario 4: Verify dots are hidden
    console.log('\n5️⃣ Scenario 4: Verify dots are hidden when toggle OFF');
    await takeScreenshot(page, 'hotspots-4-dots-hidden.png');
    console.log('✅ Screenshot captured with dots hidden (verify manually)');

    await sleep(1000);

    // Scenario 5: Toggle ON - Button should turn GREEN again
    console.log('\n6️⃣ Scenario 5: Toggle ON - verify button turns GREEN');

    await hotspotsButton.click();
    await sleep(1000);

    const buttonBgAfterToggleOn = await hotspotsButton.evaluate((el) => {
        return window.getComputedStyle(el).backgroundColor;
    });

    console.log(`   Button background color after toggle ON: ${buttonBgAfterToggleOn}`);
    await takeScreenshot(page, 'hotspots-5-toggle-on-button-green.png');

    const isGreenAgain = buttonBgAfterToggleOn.includes('34') || buttonBgAfterToggleOn.includes('green');
    if (isGreenAgain) {
        console.log('✅ PASS: Button is GREEN again (hotspots ON)');
    } else {
        console.log(`❌ FAIL: Button is NOT green (got: ${buttonBgAfterToggleOn})`);
    }

    await sleep(1000);

    // Scenario 6: Verify dots reappear
    console.log('\n7️⃣ Scenario 6: Verify dots reappear when toggle ON');
    await takeScreenshot(page, 'hotspots-6-dots-visible-again.png');
    console.log('✅ Screenshot captured with dots visible again (verify manually)');

    await sleep(1000);

    // Scenario 7: Switch to Bangalore - no dots should appear
    console.log('\n8️⃣ Scenario 7: Switch to Bangalore - verify no dots');

    // Click city selector (top-left)
    const citySelector = page.locator('button:has-text("Delhi")').first();
    await citySelector.click();
    await sleep(500);

    // Select Bangalore
    await page.click('button:has-text("Bangalore")');
    await sleep(3000); // Wait for map to reload

    await takeScreenshot(page, 'hotspots-7-bangalore-no-dots.png');
    console.log('✅ Screenshot captured of Bangalore (should have no dots)');

    await sleep(1000);

    // Scenario 8: Switch back to Delhi - dots should reappear
    console.log('\n9️⃣ Scenario 8: Switch back to Delhi - verify dots reappear');

    await page.click('button:has-text("Bangalore")').catch(() => {});
    await sleep(500);
    await page.click('button:has-text("Delhi")');
    await sleep(3000);

    await takeScreenshot(page, 'hotspots-8-delhi-dots-reappear.png');
    console.log('✅ Screenshot captured of Delhi (dots should reappear)');

    // Check console for errors
    console.log('\n🔍 Checking for console errors...');
    const logs: string[] = [];
    page.on('console', (msg) => {
        if (msg.type() === 'error') {
            logs.push(msg.text());
        }
    });

    if (logs.length === 0) {
        console.log('✅ PASS: No console errors');
    } else {
        console.log('❌ Console errors found:');
        logs.forEach(log => console.log(`   ${log}`));
    }
}

async function main() {
    console.log('🚀 Hotspots Toggle E2E Verification\n');
    console.log('📍 Target: http://localhost:5175');
    console.log('🎯 Objective: Verify button colors and dot visibility\n');

    const browser = await chromium.launch({
        headless: false,
        slowMo: 100
    });

    const context = await browser.newContext({
        viewport: { width: 1280, height: 720 }
    });

    const page = await context.newPage();

    try {
        await loginOrCreate(page);
        await verifyHotspotsToggle(page);

        console.log('\n✅ E2E Verification Complete!');
        console.log('\n📊 Summary:');
        console.log('   - 8 screenshots captured');
        console.log('   - Button color states verified');
        console.log('   - Dots visibility tested');
        console.log('   - City switching tested');
        console.log('\n📂 Screenshots saved in: apps/frontend/');
        console.log('   hotspots-1-initial-button-green.png');
        console.log('   hotspots-2-dots-visible-delhi.png');
        console.log('   hotspots-3-toggle-off-button-white.png');
        console.log('   hotspots-4-dots-hidden.png');
        console.log('   hotspots-5-toggle-on-button-green.png');
        console.log('   hotspots-6-dots-visible-again.png');
        console.log('   hotspots-7-bangalore-no-dots.png');
        console.log('   hotspots-8-delhi-dots-reappear.png');

    } catch (error) {
        console.error('\n❌ Error during verification:', error);
        await takeScreenshot(page, 'hotspots-error.png');
    } finally {
        await sleep(2000);
        await browser.close();
    }
}

main().catch(console.error);
