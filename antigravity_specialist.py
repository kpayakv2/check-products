# --- SPECIALIST CONTRACT ---
# Role: Forked Verification Specialist (Antigravity)
# Mission: Proactively identify UI/UX failures, Thai text overflows, and console errors.
# Constraint: Do NOT spawn sub-agents. Execute directly. Summarize results concisely.
# ----------------------------

import asyncio
from playwright.async_api import async_playwright
import os
import json
import time

async def check_overflow(page):
    """Deterministic check for horizontal overflows."""
    return await page.evaluate('''() => {
        const overflows = [];
        document.querySelectorAll('*').forEach(el => {
            if (el.scrollWidth > el.clientWidth && el.clientWidth > 0) {
                // Ignore script and style tags
                if (['SCRIPT', 'STYLE', 'LINK'].includes(el.tagName)) return;
                
                overflows.append({
                    tag: el.tagName,
                    id: el.id,
                    className: el.className,
                    scrollWidth: el.scrollWidth,
                    clientWidth: el.clientWidth,
                    text: el.innerText ? el.innerText.substring(0, 30) + '...' : ''
                });
            }
        });
        return overflows;
    }''')

async def run_specialist_verification(url="http://localhost:3000"):
    results = {
        "status": "PASS",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "errors": [],
        "warnings": [],
        "overflows": [],
        "screenshots": []
    }

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        # Test both Desktop and Mobile
        viewports = [
            {"name": "Desktop", "width": 1280, "height": 720},
            {"name": "Mobile", "width": 375, "height": 667}
        ]

        for vp in viewports:
            context = await browser.new_context(viewport={"width": vp["width"], "height": vp["height"]})
            page = await context.new_page()

            # Catch console errors
            page.on("console", lambda msg: 
                results["errors"].append(f"[{vp['name']}] Browser Console Error: {msg.text}") if msg.type == "error" 
                else results["warnings"].append(f"[{vp['name']}] Browser Console Warning: {msg.text}") if msg.type == "warning" else None
            )

            try:
                # Wait for network idle to ensure everything is loaded
                response = await page.goto(url, wait_until="networkidle", timeout=15000)
                if not response or response.status != 200:
                    results["status"] = "FAIL"
                    results["errors"].append(f"[{vp['name']}] Page failed to load. Status: {response.status if response else 'N/A'}")
                    continue

                # Check for overflows
                overflows = await check_overflow(page)
                if overflows:
                    results["status"] = "FAIL"
                    for o in overflows:
                        results["overflows"].append(f"[{vp['name']}] Overflow in <{o['tag']}>: {o['text']}")

                # Screenshot
                os.makedirs('output/antigravity', exist_ok=True)
                scr_path = f"output/antigravity/verify_{vp['name'].lower()}_{int(time.time())}.png"
                await page.screenshot(path=scr_path)
                results["screenshots"].append(scr_path)

            except Exception as e:
                results["status"] = "FAIL"
                results["errors"].append(f"[{vp['name']}] Exception during verification: {str(e)}")
            
            await context.close()

        await browser.close()

    # Final Summarization for Orchestrator
    print("\n--- SPECIALIST VERIFICATION SUMMARY ---")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    
    # Simple summary line for quick reading
    summary_line = f"RESULT: {results['status']} | Errors: {len(results['errors'])} | Overflows: {len(results['overflows'])} | Screenshots: {len(results['screenshots'])}"
    print(f"\n{summary_line}")

if __name__ == "__main__":
    import sys
    target_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:3000"
    asyncio.run(run_specialist_verification(target_url))
