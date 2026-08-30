import { defineConfig, devices } from '@playwright/test'
import path from 'path'

/**
 * @see https://playwright.dev/docs/test-configuration
 *
 * โปรเจกต์เดียว (chromium) โดยตั้งใจ — ชุดนี้ยิงใส่ฐานข้อมูลจริง
 * การรันข้ามเบราว์เซอร์พร้อมกันแปลว่าหลายตัวแย่งกันแก้แถวเดียวกัน
 * ผลที่ได้จึงเป็นเสียงรบกวน ไม่ใช่สัญญาณ เพิ่มเบราว์เซอร์อื่นได้เมื่อมี seed แยกต่อ worker แล้ว
 */
export default defineConfig({
  testDir: './e2e',
  // อยู่นอก testDir โดยตั้งใจ — ถ้าวางไว้ใน e2e/ Playwright จะโหลดไฟล์นี้พร้อม config
  // แล้วฟ้องว่า test.describe() ถูกเรียกจากไฟล์ที่ config import เข้ามา
  globalSetup: path.resolve(__dirname, 'playwright.setup.ts'),
  timeout: 60000,
  expect: {
    timeout: 10000,
  },

  /* Fail the build on CI if you accidentally left test.only in the source code. */
  forbidOnly: !!process.env.CI,
  /* Retry on CI only */
  retries: process.env.CI ? 2 : 0,
  /* worker เดียวเท่านั้น — ทั้งชุดใช้ฐานข้อมูลจริงตัวเดียวกัน และ dev server
     คอมไพล์หน้าเว็บตอนถูกเรียกครั้งแรก การรันขนานจึงได้ทั้งข้อมูลชนกัน
     และ timeout จากการรอคอมไพล์ ซึ่งไม่ใช่บั๊กของแอปสักอย่างเดียว
     (fullyParallel ไม่ช่วยตรงนี้ มันคุมแค่ในไฟล์เดียวกัน ไฟล์ต่างกันยังชนกันอยู่ดี) */
  workers: 1,
  /* Reporter to use. See https://playwright.dev/docs/test-reporters */
  reporter: 'html',
  /* Shared settings for all the projects below. See https://playwright.dev/docs/api/class-testoptions. */
  use: {
    /* Base URL to use in actions like `await page.goto('/')`. */
    baseURL: 'http://127.0.0.1:3000',

    /* คุกกี้ปลดล็อกที่ global-setup.ts เตรียมไว้ */
    storageState: path.resolve(__dirname, 'e2e/.auth/state.json'),

    /* Collect trace when retrying the failed test. See https://playwright.dev/docs/trace-viewer */
    trace: 'on-first-retry',

    /* Take screenshot on failure */
    screenshot: 'only-on-failure',

    /* Record video on failure */
    video: 'retain-on-failure',
  },

  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
  ],

  /* Run your local dev server before starting the tests */
  webServer: {
    command: 'npm run dev',
    url: 'http://127.0.0.1:3000',
    reuseExistingServer: !process.env.CI,
    timeout: 120 * 1000,
  },
})
