import { test, expect, type Page } from '@playwright/test'

/**
 * ตรวจ WCAG 2.2 AA ในเบราว์เซอร์จริง — ส่วนที่ __tests__/accessibility/conventions.test.ts ตรวจจากซอร์สไม่ได้
 * (docs/superpowers/plans/2026-09-19-wcag-fix-split.md หมวด "หลังแก้เสร็จ")
 *
 * อ่านอย่างเดียว: เปิดหน้า สลับแท็บ และกด Tab เท่านั้น ไม่กดปุ่มบันทึก/ลบใดๆ จึงไม่แตะฐานข้อมูลจริง
 *
 * นี่เป็นตัวแทนอัตโนมัติของการไล่ใช้คีย์บอร์ดจริง ไม่ใช่ตัวแทนที่สมบูรณ์ — มันบอกได้ว่าโฟกัสไปถึงที่ไหนและ
 * มองเห็นไหม แต่บอกไม่ได้ว่าลำดับสมเหตุสมผลหรือคนที่ใช้โปรแกรมอ่านหน้าจอจะเข้าใจหรือเปล่า
 */

const PAGES = ['/', '/products', '/unlock', '/data-quality', '/taxonomy', '/settings', '/import']

const MAX_TAB_STOPS = 40

/** ทุกมุมมองที่เปิดดูได้ในหน้า — หน้าที่มีแท็บต้องไล่ทุกแท็บ ไม่งั้นตรวจแค่แท็บแรก */
async function views(page: Page, route: string): Promise<Array<() => Promise<void>>> {
  await page.goto(route)
  await page.waitForLoadState('networkidle')
  const tabs = page.locator('[role="tab"]')
  const count = await tabs.count()
  if (count === 0) return [async () => {}]
  return Array.from({ length: count }, (_, index) => async () => {
    await tabs.nth(index).click()
    await page.waitForLoadState('networkidle')
  })
}

async function eachView(page: Page, route: string, check: (label: string) => Promise<void>) {
  const open = await views(page, route)
  for (let index = 0; index < open.length; index++) {
    await open[index]()
    await check(open.length > 1 ? `${route} [แท็บที่ ${index + 1}]` : route)
  }
}

test.describe('ไม่มี scroll แนวนอนที่ 375px', () => {
  test.use({ viewport: { width: 375, height: 812 } })

  for (const route of PAGES) {
    test(route, async ({ page }) => {
      await eachView(page, route, async (label) => {
        const overflow = await page.evaluate(() => {
          const width = window.innerWidth
          const wide = [...document.querySelectorAll('body *')]
            .filter((el) => !el.closest('nextjs-portal'))
            .map((el) => ({ el, rect: el.getBoundingClientRect() }))
            .filter(({ rect }) => rect.width > 0 && rect.right > width + 1)
            .slice(0, 5)
            .map(({ el, rect }) => `${el.tagName.toLowerCase()}.${String(el.className).slice(0, 50)} (ขวาสุด ${Math.round(rect.right)}px)`)
          return { page: document.documentElement.scrollWidth - width, wide }
        })
        expect(overflow.page, `${label}: หน้าเกินจอ ${overflow.page}px — ${overflow.wide.join(' | ')}`).toBeLessThanOrEqual(1)
      })
    })
  }
})

test.describe('ข้อความที่แสดงอยู่ไม่เล็กกว่า 12px', () => {
  for (const route of PAGES) {
    test(route, async ({ page }) => {
      await eachView(page, route, async (label) => {
        const small = await page.evaluate(() => {
          const found: string[] = []
          const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT)
          for (let node = walker.nextNode(); node; node = walker.nextNode()) {
            const text = (node.textContent || '').trim()
            const el = node.parentElement
            if (!text || !el || el.closest('nextjs-portal, script, style, noscript')) continue
            const style = getComputedStyle(el)
            const rect = el.getBoundingClientRect()
            if (style.visibility === 'hidden' || style.display === 'none' || Number(style.opacity) === 0) continue
            if (rect.width === 0 || rect.height === 0) continue
            const size = parseFloat(style.fontSize)
            if (size < 12) found.push(`${size}px "${text.slice(0, 40)}"`)
          }
          return [...new Set(found)].slice(0, 8)
        })
        expect(small, `${label}: ข้อความที่เล็กกว่า 12px`).toEqual([])
      })
    })
  }
})

test.describe('กด Tab ไล่ทั้งหน้า', () => {
  for (const route of PAGES) {
    test(route, async ({ page }) => {
      await eachView(page, route, async (label) => {
        await page.evaluate(() => (document.activeElement as HTMLElement | null)?.blur())
        const problems: string[] = []

        for (let stop = 1; stop <= MAX_TAB_STOPS; stop++) {
          await page.keyboard.press('Tab')
          const sample = () => page.evaluate(() => {
            const el = document.activeElement as HTMLElement | null
            if (!el || el === document.body) return null
            const style = getComputedStyle(el)
            const rect = el.getBoundingClientRect()
            const ring =
              (style.outlineStyle !== 'none' && parseFloat(style.outlineWidth) > 0) ||
              style.boxShadow !== 'none'
            const name = (el.getAttribute('aria-label') || el.innerText || el.getAttribute('placeholder') || '')
              .trim()
              .slice(0, 30)
            return {
              tag: `${el.tagName.toLowerCase()}${el.id ? '#' + el.id : ''}`,
              name,
              visible:
                rect.width > 0 &&
                rect.height > 0 &&
                style.visibility !== 'hidden' &&
                Number(style.opacity) > 0,
              ring,
              nextjs: !!el.closest('nextjs-portal'),
            }
          })

          // ปุ่มที่ fade-in เมื่อได้โฟกัส (CSS transition หรือ framer-motion) จะอ่านได้ opacity 0
          // ในเฟรมแรกทั้งที่ผู้ใช้จะเห็นมันในอีกไม่กี่ร้อยมิลลิวินาที — รอให้นิ่งก่อนค่อยตัดสินว่ามองไม่เห็น
          // ไม่งั้นจะรายงานปุ่มที่ใช้ได้ปกติว่าเสีย
          let focus = await sample()
          if (focus && (!focus.visible || !focus.ring)) {
            await page.waitForTimeout(600)
            focus = await sample()
          }
          if (!focus || focus.nextjs) continue
          if (!focus.visible) problems.push(`โฟกัสไปที่ของที่มองไม่เห็น: ${focus.tag} "${focus.name}"`)
          else if (!focus.ring) problems.push(`ไม่มีกรอบโฟกัส: ${focus.tag} "${focus.name}"`)
        }

        expect(problems, `${label}`).toEqual([])
      })
    })
  }
})

test.describe('เคารพ prefers-reduced-motion', () => {
  for (const route of PAGES) {
    test(route, async ({ page }) => {
      // ใช้ emulateMedia แทน test.use({ reducedMotion }) — option นั้นไม่อยู่ใน type ของรุ่นที่ติดตั้ง
      // และถ้าถูกเมินเงียบๆ เทสต์จะรันโดยไม่ได้จำลอง reduced-motion จริง แล้วผลที่ได้ใช้ไม่ได้
      await page.emulateMedia({ reducedMotion: 'reduce' })
      await eachView(page, route, async (label) => {
        const emulated = await page.evaluate(() => matchMedia('(prefers-reduced-motion: reduce)').matches)
        expect(emulated, `${label}: การจำลอง reduced-motion ไม่มีผล ผลที่ได้จึงใช้ไม่ได้`).toBe(true)

        const looping = await page.evaluate(() =>
          [...document.querySelectorAll('body *')]
            .filter((el) => !el.closest('nextjs-portal'))
            .filter((el) => {
              const s = getComputedStyle(el)
              // animate-spin คือตัวบอกว่ากำลังโหลด ถือเป็นการเคลื่อนไหวที่จำเป็น จึงไม่นับ
              return s.animationName !== 'none' && s.animationIterationCount === 'infinite' && !/spin/.test(s.animationName)
            })
            .slice(0, 5)
            .map((el) => `${el.tagName.toLowerCase()}.${String(el.className).slice(0, 60)}`)
        )
        expect(looping, `${label}: ยังมีแอนิเมชันวนไม่หยุด`).toEqual([])
      })
    })
  }
})
