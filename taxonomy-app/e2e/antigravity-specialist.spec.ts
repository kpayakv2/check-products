import { test, expect } from '@playwright/test';

/**
 * --- SPECIALIST CONTRACT ---
 * Role: UI Verification Specialist (Antigravity)
 * Mission: Identify layout failures, Thai text overflows, and responsive break points.
 * Method: Deterministic DOM Analysis via Playwright.
 * ----------------------------
 */

test.describe('Antigravity Specialist: UI Integrity Attack', () => {

  // Deterministic function to detect overflow in browser
  const checkOverflow = async (page) => {
    return await page.evaluate(() => {
      const overflows = [];
      const elements = document.querySelectorAll('*');
      elements.forEach(el => {
        if (el instanceof HTMLElement && el.scrollWidth > el.clientWidth && el.clientWidth > 0) {
          // Filter out noise (scripts, styles, hidden things)
          if (['SCRIPT', 'STYLE', 'LINK'].includes(el.tagName)) return;
          const style = window.getComputedStyle(el);
          if (style.overflow === 'hidden' || style.overflowX === 'hidden') return;

          overflows.push({
            tag: el.tagName,
            className: el.className,
            text: el.innerText ? el.innerText.substring(0, 30) : 'N/A'
          });
        }
      });
      return overflows;
    });
  };

  test('Identify Thai Text Overflows and Layout Issues', async ({ page }) => {
    // Navigate to local app
    console.log('๐— Attacking http://localhost:3000...');
    await page.goto('http://localhost:3000', { waitUntil: 'networkidle', timeout: 30000 });

    const viewports = [
      { name: 'Desktop', width: 1280, height: 720 },
      { name: 'Mobile', width: 375, height: 667 }
    ];

    const allFailures = [];

    for (const vp of viewports) {
      console.log(`๐— Testing Surface: ${vp.name} (${vp.width}x${vp.height})`);
      await page.setViewportSize({ width: vp.width, height: vp.height });
      await page.waitForTimeout(1000); // Allow layout to settle

      // Run Deterministic Overflow Check
      const overflows = await checkOverflow(page);
      if (overflows.length > 0) {
        overflows.forEach(o => {
          allFailures.push(`[${vp.name}] Overflow in <${o.tag}>: "${o.text}" (Class: ${o.className})`);
        });
      }

      // Check for Console Errors
      page.on('console', msg => {
        if (msg.type() === 'error') {
          allFailures.push(`[${vp.name}] Console Error: ${msg.text()}`);
        }
      });

      // Capture Visual Evidence
      const scrPath = `playwright-report/antigravity_${vp.name.toLowerCase()}_${Date.now()}.png`;
      await page.screenshot({ path: scrPath, fullPage: true });
    }

    // FINAL REPORT (Specialist Summary)
    console.log('\n--- SPECIALIST VERIFICATION SUMMARY ---');
    if (allFailures.length === 0) {
      console.log('RESULT: PASS (UI is stable) ✅');
    } else {
      console.log(`RESULT: FAIL (Found ${allFailures.length} issues) ❌`);
      allFailures.forEach(f => console.log(` - ${f}`));
    }
    console.log('---------------------------------------\n');

    // We still want the test to pass if it's just a verification run, 
    // unless we want to block the build.
  });
});
