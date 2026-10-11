import { NextRequest, NextResponse } from 'next/server'
import { z } from 'zod'
import { createRow, insertRows } from '@/utils/admin-db'
import { withErrorHandling } from '@/utils/error-handler'

const CreateSynonymTermSchema = z.object({
  term: z.string().min(1, 'คำพ้องต้องไม่ว่าง'),
  is_primary: z.boolean().optional(),
  confidence_score: z.number().min(0).max(1).optional(),
  source: z.string().optional(),
  language: z.string().optional(),
})

const UUID_REGEX = /^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$/

const CreateSynonymSchema = z.object({
  name_th: z.string().min(1).optional(),
  name: z.string().min(1).optional(),
  code: z.string().optional(),
  description: z.string().nullable().optional(),
  category_id: z.string().regex(UUID_REGEX, 'category_id ต้องเป็น UUID').nullable().optional(),
  terms: z.array(CreateSynonymTermSchema).min(1, 'ต้องมีคำพ้องอย่างน้อย 1 คำ'),
}).refine((data) => Boolean(data.name_th || data.name), {
  message: 'ต้องมีชื่อ synonym (name_th หรือ name)',
  path: ['name_th'],
})

/**
 * POST /api/synonyms
 *
 * เขียนผ่าน service role — `synonym_lemmas`/`synonym_terms` ให้เฉพาะ
 * taxonomy_editor/admin เขียนได้ anon จะถูก RLS ปฏิเสธ
 *
 * เดิม route นี้บังคับให้ส่ง `terms` มาแล้ว **ไม่เคยบันทึก terms เลย** —
 * สร้างแต่ตัว lemma เปล่า ๆ คำพ้องที่ผู้ใช้พิมพ์หายไปเงียบ ๆ
 */
export async function POST(request: NextRequest): Promise<NextResponse> {
  return withErrorHandling(async () => {
    let body: unknown
    try {
      body = await request.json()
    } catch {
      return NextResponse.json(
        { success: false, error: 'Invalid JSON body' },
        { status: 400 }
      )
    }

    const validation = CreateSynonymSchema.safeParse(body)
    if (!validation.success) {
      return NextResponse.json(
        { success: false, error: 'ต้องมีชื่อ synonym และคำพ้องอย่างน้อย 1 คำ', details: validation.error.issues },
        { status: 400 }
      )
    }

    const { name, name_th, code, description, category_id, terms } = validation.data
    const lemmaName = name_th || name!

    const synonym = await createRow<{ id: string }>('synonym_lemmas', {
      code: code || `SYN-${Date.now().toString(36).toUpperCase()}`,
      name_th: lemmaName,
      description: description ?? null,
      category_id: category_id ?? null,
      is_active: true
    })

    const savedTerms = await insertRows('synonym_terms', terms.map((term) => ({
      lemma_id: synonym.id,
      term: term.term,
      is_primary: term.is_primary ?? false,
      confidence_score: term.confidence_score ?? 1,
      source: term.source ?? 'manual',
      language: term.language ?? 'th',
      is_verified: (term.source ?? 'manual') === 'manual'
    })))

    return NextResponse.json(
      { success: true, data: { ...synonym, terms: savedTerms } },
      { status: 201 }
    )
  })
}
