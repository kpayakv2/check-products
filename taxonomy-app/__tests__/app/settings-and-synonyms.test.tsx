/**
 * สองหน้านี้เขียน/อ่านตารางที่ RLS ปิดไว้สำหรับ anon key
 * - /settings: `system_settings` ตอบ 406 ตอนโหลด และบันทึกไม่ลงเลย
 * - SynonymsPanel: `synonym_lemmas` / `synonym_terms` เขียนไม่ได้
 * ทั้งคู่ต้องผ่าน API route ที่ใช้ service role
 */
import { render, screen, fireEvent, waitFor } from '@testing-library/react'

jest.mock('@/components/Layout/Sidebar', () => ({ __esModule: true, default: () => null }))
jest.mock('@/components/Layout/Header', () => ({ __esModule: true, default: () => null }))
jest.mock('react-hot-toast', () => ({ toast: { error: jest.fn(), success: jest.fn() } }))

jest.mock('@/utils/supabase', () => ({
  DatabaseService: {
    getRegexRules: jest.fn(async () => []),
    getKeywordRules: jest.fn(async () => []),
    getSynonyms: jest.fn(async () => []),
    getTaxonomyTree: jest.fn(async () => [])
  }
}))

import SettingsPage from '@/app/settings/page'
import SynonymsPanel from '@/components/Taxonomy/SynonymsPanel'

let fetchMock: jest.Mock

beforeEach(() => {
  fetchMock = jest.fn(async () => ({
    ok: true,
    status: 200,
    json: async () => ({ success: true, data: { id: 'row-1' } })
  }))
  global.fetch = fetchMock as unknown as typeof fetch
})

const requestsTo = (fragment: string) =>
  fetchMock.mock.calls.filter(call => String(call[0]).includes(fragment))

describe('หน้าตั้งค่า', () => {
  it('อ่านค่าตั้งค่าผ่าน API route ไม่ใช่คุยกับ Supabase ตรงด้วย anon key', async () => {
    render(<SettingsPage />)

    await waitFor(() => expect(requestsTo('/api/settings').length).toBeGreaterThan(0))
  })
})

describe('แผงคำพ้อง', () => {
  it('สร้าง synonym พร้อมคำพ้องผ่าน API route เดียว', async () => {
    render(<SynonymsPanel />)

    fireEvent.click(await screen.findByTestId('add-synonym-btn'))
    fireEvent.change(screen.getByPlaceholderText('e.g. สมาร์ทโฟน'), { target: { value: 'ปากกา' } })
    fireEvent.click(screen.getByTestId('add-term-btn'))
    fireEvent.change(screen.getAllByPlaceholderText('Term Variation')[0], {
      target: { value: 'ปากกาลูกลื่น' }
    })
    fireEvent.click(screen.getByTestId('save-synonym-btn'))

    await waitFor(() => expect(requestsTo('/api/synonyms').length).toBeGreaterThan(0))
    const [, init] = requestsTo('/api/synonyms')[0]
    const body = JSON.parse(String((init as RequestInit).body))
    expect(body.terms[0].term).toBe('ปากกาลูกลื่น')
  })
})
