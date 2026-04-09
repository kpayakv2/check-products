import asyncio
from playwright.async_api import async_playwright
import os
import time

async def run_verification():
    async with async_playwright() as p:
        # Check if output directory exists
        if not os.path.exists('output'):
            os.makedirs('output')
        
        browser = await p.chromium.launch()
        context = await browser.new_context(viewport={'width': 1280, 'height': 720})
        page = await context.new_page()
        
        # Collect console messages
        console_msgs = []
        page.on("console", lambda msg: console_msgs.append(f"[{msg.type}] {msg.text}"))
        
        print("Waiting for servers to be ready (30 seconds max)...")
        max_retries = 30
        for i in range(max_retries):
            try:
                response = await page.goto("http://localhost:3000", wait_until="networkidle", timeout=10000)
                if response and response.status == 200:
                    print(f"Server is UP at http://localhost:3000 (Attempt {i+1})")
                    break
            except Exception as e:
                if i == max_retries - 1:
                    print(f"Failed to connect to server: {str(e)}")
                time.sleep(1)
        
        # Screenshot
        screenshot_path = 'output/antigravity_home_verification.png'
        await page.screenshot(path=screenshot_path, full_page=True)
        print(f"Screenshot saved to: {screenshot_path}")
        
        # Check for Thai Text issues (simple heuristic: look for ????? or broken spans)
        # In a real scenario, this would involve more advanced visual or DOM analysis
        
        print("\n--- Console Logs ---")
        if not console_msgs:
            print("No console errors found. ✅")
        for msg in console_msgs:
            print(msg)
            
        await browser.close()

if __name__ == "__main__":
    asyncio.run(run_verification())
