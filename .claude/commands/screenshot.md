# screenshot

Automated browser screenshot tool for UI debugging and visual testing.

## What It Does
Uses Playwright to launch a browser and capture screenshots of the FloodSafe frontend at different stages of user interaction. Helps identify UI issues, layout problems, and visual regressions.

## How to Run
```bash
cd apps/frontend
npm run screenshot
```

Ensure the frontend dev server is running on `http://localhost:5175` before executing.

## Screenshots Captured
1. **Home Screen** - Initial app load
2. **Flood Atlas** - Main map view after navigation
3. **City Switch** - Delhi city selected (if city selector visible)
4. **History Panel** - Historical floods panel open
5. **Panel Closeup** - Detailed view of historical panel
6. **Scrolled View** - Panel with scrolled content (if scrollable)
7. **Error Screenshot** - Captured if any errors occur

## Output
Screenshots are saved to `apps/frontend/` with format: `screenshot-{step}-{name}.png`

## For Debugging UI Issues
- Compare screenshots across commits to spot visual regressions
- Check panel dimensions logged in console for responsive design issues
- Review error screenshots if the script fails (look at browser console errors)
- Verify overlay scrollbars are disabled for consistent screenshots
- Adjust viewport (1400x900) if testing specific mobile breakpoints

## Browser Configuration
- Headless: false (visible window)
- Viewport: 1400x900
- Overlay scrollbars: disabled
- Timeout: 30 seconds max per operation
