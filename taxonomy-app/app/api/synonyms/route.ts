import { NextRequest, NextResponse } from 'next/server'
import { DatabaseService } from '@/utils/supabase'
import { createRow, insertRows } from '@/utils/admin-db'
import { withErrorHandling } from '@/utils/error-handler'

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url)
    const categoryId = searchParams.get('category_id')

    const synonyms = await DatabaseService.getSynonyms(categoryId || undefined)
    return NextResponse.json({ success: true, data: synonyms })
  } catch (error) {
    console.error('Error fetching synonyms:', error)
    return NextResponse.json(
      { success: false, error: 'Failed to fetch synonyms' },
      { status: 500 }
    )
  }
}

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
    const body = await request.json()
    const { name, name_th, code, description, category_id, terms } = body
    const lemmaName = name_th || name

    if (!lemmaName || !Array.isArray(terms) || terms.length === 0) {
      return NextResponse.json(
        { success: false, error: 'ต้องมีชื่อ synonym และคำพ้องอย่างน้อย 1 คำ' },
        { status: 400 }
      )
    }

    const synonym = await createRow<{ id: string }>('synonym_lemmas', {
      code: code || `SYN-${Date.now().toString(36).toUpperCase()}`,
      name_th: lemmaName,
      description,
      category_id,
      is_active: true
    })

    const savedTerms = await insertRows('synonym_terms', terms.map((term: Record<string, unknown>) => ({
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
