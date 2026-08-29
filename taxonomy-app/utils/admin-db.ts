/**
 * งานเขียนข้อมูลที่ต้องใช้ service role
 *
 * `utils/supabase.ts` สร้าง client ด้วย anon key เสมอ (ทั้งฝั่งเบราว์เซอร์และฝั่ง server)
 * แต่ policy ของ `taxonomy_nodes`, `synonym_lemmas`, `synonym_terms` และ
 * `system_settings` ให้เขียนได้เฉพาะ role `taxonomy_editor` / `taxonomy_admin`
 * ผลคือ insert ถูกปฏิเสธ (42501) ส่วน update/delete "สำเร็จ" แบบไม่แตะข้อมูลเลย
 * เพราะ RLS กรองแถวทิ้งก่อน — เงียบสนิท ไม่มี error ให้จับ
 *
 * ทุกฟังก์ชันในไฟล์นี้จึงเรียกผ่าน `supabaseAdmin` และคืนค่าที่บอกได้ว่า
 * "ไม่พบแถว" ต่างจาก "แก้สำเร็จ" เพื่อไม่ให้ UI รายงานผลลวงอีก
 */
import { supabaseAdmin } from './supabase-admin'

type Row = Record<string, unknown>

/** ตัดคีย์ที่เป็น undefined หรือสตริงว่างออก — คอลัมน์ uuid รับ '' ไม่ได้ (22P02) */
export const omitEmpty = (input: Row): Row =>
  Object.fromEntries(
    Object.entries(input).filter(([, value]) => value !== undefined && value !== '')
  )

export async function createRow<T = Row>(table: string, input: Row): Promise<T> {
  const { data, error } = await supabaseAdmin
    .from(table)
    .insert(omitEmpty(input))
    .select()
    .single()

  if (error) throw error
  return data as T
}

/** คืน null เมื่อไม่มีแถวนั้น (หรือถูก RLS กรอง) แทนที่จะเงียบแล้วดูเหมือนสำเร็จ */
export async function updateRow<T = Row>(table: string, id: string, patch: Row): Promise<T | null> {
  const { data, error } = await supabaseAdmin
    .from(table)
    .update(omitEmpty(patch))
    .eq('id', id)
    .select()
    .maybeSingle()

  if (error) throw error
  return (data as T) ?? null
}

/** คืน false เมื่อไม่มีแถวไหนถูกลบจริง */
export async function deleteRow(table: string, id: string): Promise<boolean> {
  const { data, error } = await supabaseAdmin
    .from(table)
    .delete()
    .eq('id', id)
    .select('id')

  if (error) throw error
  return Array.isArray(data) ? data.length > 0 : Boolean(data)
}

export async function insertRows<T = Row>(table: string, rows: Row[]): Promise<T[]> {
  const { data, error } = await supabaseAdmin
    .from(table)
    .insert(rows.map(omitEmpty))
    .select()

  if (error) throw error
  return (data as T[]) ?? []
}

/**
 * แถวคอนฟิกของระบบ
 *
 * `system_settings` ปนสองแบบอยู่ในตารางเดียว: แถวคอนฟิกที่เก็บเป็น JSONB
 * (`setting_key` เป็น null) กับแถว key/value อย่าง taxonomy_version /
 * total_categories / last_updated — ตอนนี้มี 4 แถวในฐานข้อมูลจริง
 * ถ้าอ่านแบบ `limit(1)` เฉย ๆ Postgres ไม่การันตีว่าจะได้แถวไหน
 */
export async function readSettingsRow<T = Row>(): Promise<T | null> {
  const { data, error } = await supabaseAdmin
    .from('system_settings')
    .select('*')
    .is('setting_key', null)
    .order('updated_at', { ascending: true })
    .limit(1)
    .maybeSingle()

  if (error) throw error
  return (data as T) ?? null
}

/** เขียนทับแถวคอนฟิกเดิมเสมอ สร้างใหม่เฉพาะตอนที่ยังไม่มีแถวนั้นจริง ๆ */
export async function writeSettingsRow<T = Row>(input: Row): Promise<T> {
  const current = await readSettingsRow<{ id?: string }>()
  const payload = { ...input, updated_at: new Date().toISOString() }

  if (!current?.id) {
    return createRow<T>('system_settings', payload)
  }

  const { data, error } = await supabaseAdmin
    .from('system_settings')
    .update(payload)
    .eq('id', current.id)
    .select()
    .single()

  if (error) throw error
  return data as T
}

/** อ่านแถวเดียวด้วย id ผ่าน service role (ใช้ในเส้นทางเขียน เพื่อไม่ปนกับ client ของ anon) */
export async function readRowById<T = Row>(table: string, id: string): Promise<T | null> {
  const { data, error } = await supabaseAdmin
    .from(table)
    .select('*')
    .eq('id', id)
    .maybeSingle()

  if (error) throw error
  return (data as T) ?? null
}

/** ลำดับถัดไปในกลุ่มเดียวกัน (ใต้หมวดแม่เดียวกัน หรือระดับบนสุด) */
export async function nextSortOrder(parentId?: string): Promise<number> {
  let query = supabaseAdmin
    .from('taxonomy_nodes')
    .select('sort_order')
    .order('sort_order', { ascending: false })
    .limit(1)

  query = parentId ? query.eq('parent_id', parentId) : query.is('parent_id', null)

  const { data, error } = await query
  if (error) throw error

  const rows = (data as { sort_order?: number }[]) ?? []
  return rows.length > 0 ? (rows[0].sort_order ?? 0) + 1 : 0
}
