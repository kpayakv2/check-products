import { NextRequest, NextResponse } from 'next/server'
import { supabaseAdmin } from '@/utils/supabase-admin'
import { withErrorHandling } from '@/utils/error-handler'
import {
  buildCategoryChain,
  countPosExportIssues,
  toPosExportRow,
  type ExportableProduct,
  type PosExportRow,
  type TaxonomyNodeLite,
} from '@/utils/pos-export'
import { allocateItemIds } from '@/utils/pos-item-id'

const PREVIEW_LIMIT = 50

/** จำนวนแถวที่ส่งกลับให้ UI ดูก่อนตัดสินใจโหลด — ไม่ใช่จำนวนที่จะอยู่ในไฟล์จริง */
export const dynamic = 'force-dynamic'

interface ExportProductRow extends ExportableProduct {
  id: string
}

/**
 * ดึงสินค้าที่ยังไม่ได้ส่งเข้า POS พร้อมแปลงเป็นแถวของไฟล์
 *
 * คัดด้วย metadata ไม่ใช่ status เพราะ /api/verify เซ็ต status เป็น 'approved'
 * ตอนคนกดยืนยันหมวด ซึ่งเป็นค่าเดียวกับสตอกเดิม 3,103 รายการ
 * metadata.source เป็นตัวเดียวที่ยังแยก 'import_wizard' ออกจาก 'legacy_labelled' ได้
 */
async function loadExportRows(): Promise<{
  ids: string[]
  rows: PosExportRow[]
  assignedSkus: Map<string, string>
}> {
  const { data: products, error: productsError } = await supabaseAdmin
    .from('products')
    .select('id, name_th, sku, barcode, unit_name, vat_flag, price, category_id')
    .eq('metadata->>source', 'import_wizard')
    .eq('metadata->>dedup_bucket', 'new')
    .is('pos_exported_at', null)
    .order('name_th')

  if (productsError) throw productsError

  // taxonomy มีแค่ 134 โหนด ดึงมาทั้งก้อนแล้วไล่สายในหน่วยความจำ
  // ถูกกว่ายิงหา parent ทีละรายการ และเลี่ยง raw SQL ตามกฎรีโป
  const { data: nodes, error: nodesError } = await supabaseAdmin
    .from('taxonomy_nodes')
    .select('id, name_th, parent_id')

  if (nodesError) throw nodesError

  // ต้องรู้ sku ทั้งหมดที่ใช้ไปแล้ว ไม่ใช่แค่เลขสูงสุด — products.sku มี UNIQUE
  // constraint ถ้าเลขในฐานข้อมูลขาดช่วง การเดินจาก max อย่างเดียวยังชนได้
  const { data: usedSkus, error: skuError } = await supabaseAdmin
    .from('products')
    .select('sku')
    .not('sku', 'is', null)

  if (skuError) throw skuError

  const nodesById = new Map<string, TaxonomyNodeLite>(
    ((nodes ?? []) as TaxonomyNodeLite[]).map((node) => [node.id, node])
  )

  const list = (products ?? []) as ExportProductRow[]

  // แจกรหัสทั้งใน GET และ POST ด้วยข้อมูลชุดเดียวกัน ต่างกันแค่ POST เขียนลงฐานข้อมูล
  // ตัวแจกเรียงตามชื่อเสมอ รหัสที่เห็นในหน้าตัวอย่างจึงตรงกับไฟล์ที่โหลดจริง
  const assignedSkus = allocateItemIds(
    list.map((product) => ({
      id: product.id,
      name_th: product.name_th,
      sku: product.sku,
      mainCategory: buildCategoryChain(product.category_id, nodesById)[0] ?? null,
    })),
    ((usedSkus ?? []) as { sku: string | null }[])
      .map((row) => row.sku ?? '')
      .filter(Boolean)
  )

  return {
    ids: list.map((product) => product.id),
    rows: list.map((product) =>
      toPosExportRow(
        { ...product, sku: product.sku ?? assignedSkus.get(product.id) ?? null },
        nodesById
      )
    ),
    assignedSkus,
  }
}

export { loadExportRows }

/**
 * GET /api/pos-export
 * ยอดรวม + จำนวนช่องที่ยังขาด + ตัวอย่างแถวแรก ๆ ให้ดูก่อนกดโหลด
 */
export async function GET(_request: NextRequest): Promise<NextResponse> {
  return withErrorHandling(async () => {
    const { rows } = await loadExportRows()

    return NextResponse.json({
      success: true,
      total: rows.length,
      issues: countPosExportIssues(rows),
      preview: rows.slice(0, PREVIEW_LIMIT),
    })
  })
}
