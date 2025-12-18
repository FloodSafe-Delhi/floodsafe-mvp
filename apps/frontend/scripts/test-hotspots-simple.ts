/**
 * Simple Hotspots Toggle Test
 *
 * Prerequisites: User must be logged in and on HomeScreen (Delhi)
 * Run: npx tsx scripts/test-hotspots-simple.ts
 */

import { chromium } from 'playwright';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const BASE_URL = 'http://localhost:5175';
const SCREENSHOT_DIR = path.join(__dirname, '..');

async function sleep(ms: number) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

async function main() {
    console.log('🚀 Simple Hotspots Toggle Test\n');
    console.log('⚠️  Make sure you are logged in and on HomeScreen\n');

    const browser = await chromium.launch({
        headless: false,
        slowMo: 500
    });

    const context = await browser.newContext({
        viewport: { width: 1280, height: 720 }
    });

    const page = await context.newPage();

    try {
        console.log('1️⃣ Navigating to app...');
        await page.goto(BASE_URL);
        await sleep(5000); // Wait for app to load

        console.log('\n2️⃣ Looking for hotspots button...');

        // Find the hotspots button with Droplets icon
        const hotspotsButton = page.locator('button[title*="waterlogging hotspots"]');

        // Wait for button to be visible
        await hotspotsButton.waitFor({ state: 'visible', timeout: 10000 });
        console.log('✓ Button found!');

        // Get initial button color
        console.log('\n3️⃣ Checking button color...');
        const initialBg = await hotspotsButton.evaluate((el) => {
            const style = window.getComputedStyle(el);
            return {
                bg: style.backgroundColor,
                classes: el.className
            };
        });

        console.log(`   Background: ${initialBg.bg}`);
        console.log(`   Classes: ${initialBg.classes}`);

        // Screenshot 1: Initial state
        await page.screenshot({
            path: path.join(SCREENSHOT_DIR, 'hotspots-1-initial.png'),
            fullPage: false
        });
        console.log('📸 Screenshot 1: Initial state saved');

        await sleep(2000);

        // Click to toggle OFF
        console.log('\n4️⃣ Clicking button to toggle OFF...');
        await hotspotsButton.click();
        await sleep(1500);

        const afterToggleOffBg = await hotspotsButton.evaluate((el) => {
            return window.getComputedStyle(el).backgroundColor;
        });
        console.log(`   Background after toggle OFF: ${afterToggleOffBg}`);

        // Screenshot 2: Toggled OFF
        await page.screenshot({
            path: path.join(SCREENSHOT_DIR, 'hotspots-2-toggle-off.png'),
            fullPage: false
        });
        console.log('📸 Screenshot 2: Toggled OFF saved');

        await sleep(2000);

        // Click to toggle ON
        console.log('\n5️⃣ Clicking button to toggle ON...');
        await hotspotsButton.click();
        await sleep(1500);

        const afterToggleOnBg = await hotspotsButton.evaluate((el) => {
            return window.getComputedStyle(el).backgroundColor;
        });
        console.log(`   Background after toggle ON: ${afterToggleOnBg}`);

        // Screenshot 3: Toggled ON
        await page.screenshot({
            path: path.join(SCREENSHOT_DIR, 'hotspots-3-toggle-on.png'),
            fullPage: false
        });
        console.log('📸 Screenshot 3: Toggled ON saved');

        // Check for console errors
        console.log('\n6️⃣ Checking console...');
        const consoleLogs: string[] = [];
        page.on('console', msg => {
            if (msg.type() === 'error') {
                consoleLogs.push(msg.text());
            }
        });

        await sleep(2000);

        if (consoleLogs.length === 0) {
            console.log('✅ No console errors');
        } else {
            console.log('❌ Console errors:');
            consoleLogs.forEach(log => console.log(`   ${log}`));
        }

        console.log('\n✅ Test Complete!');
        console.log('\n📊 Results:');
        console.log(`   Initial state: ${initialBg.bg}`);
        console.log(`   After toggle OFF: ${afterToggleOffBg}`);
        console.log(`   After toggle ON: ${afterToggleOnBg}`);
        console.log('\n📂 Screenshots saved:');
        console.log('   - hotspots-1-initial.png');
        console.log('   - hotspots-2-toggle-off.png');
        console.log('   - hotspots-3-toggle-on.png');

    } catch (error) {
        console.error('\n❌ Error:', error);
        await page.screenshot({
            path: path.join(SCREENSHOT_DIR, 'hotspots-error-simple.png')
        });
    } finally {
        await sleep(3000);
        await browser.close();
    }
}

main().catch(console.error);
