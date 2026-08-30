import { request, type FullConfig } from '@playwright/test'
import { config as loadEnv } from 'dotenv'
import fs from 'fs'
import path from 'path'

/**
 * ปลดล็อกเซสชันหนึ่งครั้งก่อนรันทั้งชุด แล้วเก็บคุกกี้ไว้ให้ทุก spec ใช้ร่วมกัน
 *
 * middleware.ts กั้นทุกคำขอที่ไม่ใช่ GET ไว้หลังคุกกี้ `internal_session`
 * ถ้าไม่ปลดล็อกก่อน ทุก spec ที่กดบันทึก/ลบจะได้ 401 แล้วล้มโดยไม่เกี่ยวกับ UI เลย
 * ซึ่งเป็นคนละอาการกับที่เราอยากให้ชุดนี้จับ
 */
async function globalSetup(config: FullConfig) {
  loadEnv({ path: path.resolve(__dirname, '.env.local') })

  const secret = process.env.INTERNAL_API_SECRET
  if (!secret) {
    throw new Error(
      'ไม่พบ INTERNAL_API_SECRET ใน taxonomy-app/.env.local — ' +
        'ชุด e2e ปลดล็อกเซสชันไม่ได้ และทุกการบันทึกจะได้ 401'
    )
  }

  const baseURL = config.projects[0]?.use?.baseURL ?? 'http://127.0.0.1:3000'
  const authDir = path.resolve(__dirname, 'e2e', '.auth')
  const statePath = path.join(authDir, 'state.json')

  const api = await request.newContext({ baseURL })
  const response = await api.post('/api/unlock', { data: { secret } })
  if (!response.ok()) {
    throw new Error(
      `ปลดล็อกไม่สำเร็จ (${response.status()}): ${await response.text()}`
    )
  }

  fs.mkdirSync(authDir, { recursive: true })
  await api.storageState({ path: statePath })
  await api.dispose()
}

export default globalSetup
