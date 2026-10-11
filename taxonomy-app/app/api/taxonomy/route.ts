import { NextRequest, NextResponse } from 'next/server'
import { z } from 'zod'
import { createRow, nextSortOrder, readRowById } from '@/utils/admin-db'
import { rateLimit } from '@/utils/rate-limit'
import { withErrorHandling } from '@/utils/error-handler'
import { validateRequest } from '@/utils/validation'
import { logger } from '@/utils/logger'

import type { TaxonomyNode, ApiResponse } from '@/types'


// Validation Schemas
const CreateTaxonomySchema = z.object({
  name_th: z.string()
    .min(1, 'ชื่อภาษาไทยต้องไม่ว่าง')
    .max(255, 'ชื่อภาษาไทยต้องไม่เกิน 255 ตัวอักษร')
    .regex(/[\u0e00-\u0e7f]/, 'ต้องมีอักษรไทยอย่างน้อย 1 ตัว'),
  name_en: z.string()
    .max(255, 'ชื่อภาษาอังกฤษต้องไม่เกิน 255 ตัวอักษร')
    .optional(),
  description: z.string()
    .max(1000, 'คำอธิบายต้องไม่เกิน 1000 ตัวอักษร')
    .optional(),
  // ฟอร์มส่งสตริงว่างมาเมื่อไม่ได้เลือกหมวดแม่ ต้องกลายเป็น undefined
  // ไม่ใช่ '' ที่ทำให้คอลัมน์ uuid ตอบ 22P02 invalid input syntax
  parent_id: z.preprocess(
    (value) => (value === '' || value === null ? undefined : value),
    z.string().uuid('parent_id ต้องเป็น UUID ที่ถูกต้อง').optional()
  ),
  keywords: z.array(z.string())
    .max(20, 'keywords ต้องไม่เกิน 20 รายการ')
    .optional(),
  metadata: z.record(z.string(), z.any())
    .optional()
})

// Rate limiting configuration
const limiter = rateLimit({
  interval: 60 * 1000, // 1 minute
  uniqueTokenPerInterval: 500, // Max 500 unique IPs per minute
})

/**
 * POST /api/taxonomy
 * สร้าง taxonomy node ใหม่
 */
export async function POST(request: NextRequest): Promise<NextResponse> {
  return withErrorHandling(async () => {
    // Rate limiting - stricter for write operations
    try {
      await limiter.check(5, 'taxonomy-post') // 5 requests per minute per IP
    } catch {
      return NextResponse.json(
        { success: false, error: 'Too many requests' },
        { status: 429 }
      )
    }

    // Parse and validate request body
    const body = await validateRequest(request, CreateTaxonomySchema)
    if ('error' in body) {
      return NextResponse.json(body, { status: 400 })
    }

    const { name_th, name_en, description, parent_id, keywords, metadata } = body

    console.info('Creating taxonomy node', { name_th, name_en, parent_id })

    // Calculate level and path
    let level = 0
    let path = '/'
    
    if (parent_id) {
      const parent = await readRowById<TaxonomyNode>('taxonomy_nodes', parent_id)
      if (!parent) {
        console.warn('Parent taxonomy node not found', { parent_id })
        return NextResponse.json(
          { success: false, error: 'Parent category not found' },
          { status: 404 }
        )
      }
      level = (parent.level ?? 0) + 1
      path = `${parent.path}${parent.id}/`
    }

    // Get next sort order
    const sort_order = await nextSortOrder(parent_id)

    // เขียนผ่าน service role — anon key ถูก RLS ปฏิเสธ (42501)
    // `code` เป็น NOT NULL และไม่มี default จึงต้องสร้างที่ฝั่ง server
    const category = await createRow<TaxonomyNode>('taxonomy_nodes', {
      code: `CAT-${Date.now().toString(36).toUpperCase()}`,
      name_th,
      name_en,
      description,
      parent_id,
      level,
      sort_order,
      path,
      keywords: keywords || [],
      metadata: metadata || {},
      is_active: true
    })

    console.info('Successfully created taxonomy node', { 
      id: category.id, 
      name_th: category.name_th 
    })

    return NextResponse.json({
      success: true,
      data: category,
      message: 'Taxonomy node created successfully'
    } as ApiResponse<TaxonomyNode>, { status: 201 })

  }, 'POST /api/taxonomy')
}
