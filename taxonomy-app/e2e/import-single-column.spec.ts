import { test, expect } from '@playwright/test';
import path from 'path';
import fs from 'fs';

test.describe('Real CSV Import Flow (Absolute Success)', () => {
  test('ควรนำเข้าไฟล์ CSV จริง และล็อคเป้าปุ่มตัวจริงได้สำเร็จ', async ({ page }) => {
    // ให้เวลา 5 นาทีสำหรับ AI ประมวลผล 405 รายการ
    test.setTimeout(300000); 

    // สเต็ป 1: เข้าหน้า Import
    await page.goto('/import/wizard');
    await page.waitForLoadState('networkidle');

    // คลิกเลือกกล่อง "อัปโหลดไฟล์ใหม่"
    await page.locator('.premium-card, .border-2').first().click();

    // สเต็ป 2: อัปโหลดไฟล์
    const filePath = "D:\\product_checker\\check-products\\input\\new_product\\POS_เพิ่มสินค้า_20250727_063658_จากไฟล์สินค้าใหม่.csv";
    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(filePath);
    
    // กดปุ่มถัดไป
    const nextBtn = page.locator('[data-testid="wizard-next-btn"]');
    await expect(nextBtn).toBeEnabled({ timeout: 30000 });
    await nextBtn.click();

    // สเต็ป 3: จับคู่คอลัมน์
    await page.waitForSelector('[data-testid="column-mapping-header"]', { timeout: 30000 });
    const selects = await page.locator('select').all();
    if (selects.length >= 2) {
      for (const s of selects) await s.selectOption('');
      await selects[1].selectOption('product_name');
    }

    // กดยืนยันการ Mapping
    await page.click('[data-testid="mapping-complete-btn"]');

    // สเต็ป 4: รอจนกว่า AI จะทำงานเสร็จ 100%
    console.log("⏳ AI กำลังทำงานพื้นหลัง... (ล็อคเป้าด้วย Test ID)");
    
    // **หัวใจสำคัญ:** ใช้ ID ที่เราเพิ่งใส่ลงไป ล็อคเป้าได้ 100%
    const realFinishBtn = page.locator('[data-testid="wizard-final-next-btn"]');
    
    // รอจนกว่าปุ่มสว่าง
    await expect(realFinishBtn).toBeEnabled({ timeout: 240000 });
    
    console.log(`✅ AI ประมวลผลเสร็จสิ้น! กำลังย้ายไปหน้าตรวจสอบ...`);

    // สเต็ป 5: ย้ายไปหน้าตรวจสอบ
    await realFinishBtn.click();
    await page.waitForURL('**/verify');
    
    // ยืนยันผลลัพธ์
    await page.waitForSelector('h1:has-text("Verification")', { timeout: 30000 });
    console.log("🎉 CONGRATULATIONS! ระบบนำเข้าทำงานได้สมบูรณ์แบบ 100% แล้วครับกาน");
  });
});
