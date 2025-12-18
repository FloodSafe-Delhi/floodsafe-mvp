import { chromium } from '@playwright/test';

async function takeScreenshots() {
  console.log('Launching browser...');
  const browser = await chromium.launch({
    headless: false,
    args: [
      '--disable-overlay-scrollbar',
      '--enable-features=OverlayScrollbar:Disabled',
      '--disable-web-security',
      '--disable-features=CrossOriginOpenerPolicy'
    ]
  });
  const page = await browser.newPage({ viewport: { width: 1400, height: 900 } });

  // Capture errors
  page.on('console', msg => {
    if (msg.type() === 'error') console.log('Browser error:', msg.text());
  });

  let step = 1;
  const screenshot = async (name: string) => {
    const filename = `screenshot-${step++}-${name}.png`;
    await page.screenshot({ path: filename });
    console.log(`Saved: ${filename}`);
  };

  try {
    // Step 1: Navigate to app
    console.log('Step 1: Navigate to app...');
    await page.goto('http://localhost:5175', { waitUntil: 'load', timeout: 30000 });
    await page.waitForTimeout(2000);
    await screenshot('home');

    // Step 2: Click Flood Atlas
    console.log('Step 2: Click Flood Atlas...');
    await page.click('text=Flood Atlas');
    await page.waitForTimeout(4000);
    await screenshot('flood-atlas');

    // Step 3: Switch to Delhi using native select
    console.log('Step 3: Switch to Delhi...');
    const citySelector = page.locator('#city-selector');
    if (await citySelector.isVisible({ timeout: 2000 }).catch(() => false)) {
      console.log('Found city selector, selecting Delhi...');
      await citySelector.selectOption('delhi');
      // Wait longer for city change to complete
      await page.waitForTimeout(5000);
      await screenshot('switched-to-delhi');
    } else {
      console.log('City selector not found');
    }

    // Step 4: Verify hotspots layer
    console.log('Step 4: Verify hotspots layer...');
    await page.waitForTimeout(3000); // Wait for hotspots API

    // 4a: Capture map with hotspots visible
    await screenshot('hotspots-visible');

    // 4b: Click on a hotspot marker (center of map where Delhi clusters)
    console.log('Step 4b: Click hotspot marker...');
    const mapCanvas = page.locator('canvas').first();
    await mapCanvas.click({ position: { x: 700, y: 400 } });
    await page.waitForTimeout(1500);
    await screenshot('hotspot-popup-open');

    // 4c: Capture popup closeup
    const popup = page.locator('.maplibregl-popup-content').first();
    if (await popup.isVisible({ timeout: 2000 }).catch(() => false)) {
      await popup.screenshot({ path: `screenshot-${step++}-popup-fhi-details.png` });
      console.log('Captured FHI popup details');
    }

    // 4d: Close popup by clicking elsewhere
    await mapCanvas.click({ position: { x: 100, y: 100 } });
    await page.waitForTimeout(500);

    // 4e: Toggle hotspots layer OFF
    console.log('Step 4e: Toggle hotspots OFF...');
    const hotspotsBtn = page.locator('button[title*="waterlogging"]');
    if (await hotspotsBtn.isVisible({ timeout: 2000 }).catch(() => false)) {
      await hotspotsBtn.click();
      await page.waitForTimeout(1000);
      await screenshot('hotspots-toggled-off');

      // 4f: Toggle back ON
      await hotspotsBtn.click();
      await page.waitForTimeout(1000);
      await screenshot('hotspots-toggled-on');
    }

    // Step 5: Click History button
    console.log('Step 5: Click History button...');
    const historyBtn = page.locator('button[title*="historical"], button[title*="History"]').first();
    if (await historyBtn.isVisible({ timeout: 2000 }).catch(() => false)) {
      await historyBtn.click();
      await page.waitForTimeout(1000);
      await screenshot('history-panel-open');

      // Step 6: Take closeup of panel
      console.log('Step 6: Capture panel closeup...');
      const panelBox = page.locator('.bg-white.rounded-xl').first();
      if (await panelBox.isVisible({ timeout: 1000 }).catch(() => false)) {
        await panelBox.screenshot({ path: `screenshot-${step++}-panel-closeup.png` });
        console.log('Saved panel closeup');

        // Get panel dimensions
        const box = await panelBox.boundingBox();
        console.log('Panel dimensions:', box);
      }

      // Step 7: Try scrolling
      console.log('Step 7: Try scrolling...');
      const scrollArea = page.locator('.custom-scrollbar, .overflow-y-auto').first();
      if (await scrollArea.isVisible({ timeout: 1000 }).catch(() => false)) {
        await scrollArea.evaluate((el) => { el.scrollTop = 200; });
        await page.waitForTimeout(500);
        await screenshot('scrolled');
      }
    } else {
      console.log('History button not found');
      await screenshot('no-history-button');
    }

    console.log('Done!');
  } catch (error) {
    console.error('Error:', error);
    await screenshot('error');
  } finally {
    await browser.close();
  }
}

takeScreenshots();
