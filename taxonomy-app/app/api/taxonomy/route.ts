import { NextRequest, NextResponse } from 'next/server'
import { z } from 'zod'
import { DatabaseService } from '@/utils/supabase'
import { rateLimit } from '@/utils/rate-limit'
import { withErrorHandling } from '@/utils/error-handler'
import { validateRequest } from '@/utils/validation'
import { logger } from '@/utils/logger'

// Types
interface TaxonomyNode {
  id: string
  name_th: string
  name_en?: string
  description?: string
  parent_id?: string
  level: number
  sort_order: number
  path?: string
  keywords?: string[]
  metadata?: Record<string, any>
  is_active: boolean
  created_at: string
  updated_at: string
}

interface ApiResponse<T = any> {
  success: boolean
  data?: T
  error?: string
  message?: string
}

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
  parent_id: z.string()
    .uuid('parent_id ต้องเป็น UUID ที่ถูกต้อง')
    .optional(),
  keywords: z.array(z.string())
    .max(20, 'keywords ต้องไม่เกิน 20 รายการ')
    .optional(),
  metadata: z.record(z.string(), z.any())
    .optional()
})

const GetTaxonomySchema = z.object({
  parent_id: z.string().uuid().optional(),
  level: z.coerce.number().min(0).max(10).optional(),
  include_inactive: z.coerce.boolean().optional(),
  limit: z.coerce.number().min(1).max(1000).default(100),
  offset: z.coerce.number().min(0).default(0)
})

// Rate limiting configuration
const limiter = rateLimit({
  interval: 60 * 1000, // 1 minute
  uniqueTokenPerInterval: 500, // Max 500 unique IPs per minute
})

/**
 * GET /api/taxonomy
 * ดึงข้อมูล taxonomy tree หรือ nodes ตามเงื่อนไข
 */
export async function GET(request: NextRequest): Promise<NextResponse> {
  return withErrorHandling(async () => {
    // Rate limiting
    try {
      await limiter.check(10, 'taxonomy-get') // 10 requests per minute per IP
    } catch {
      return NextResponse.json(
        { success: false, error: 'Too many requests' },
        { status: 429 }
      )
    }

    // Parse and validate query parameters
    const { searchParams } = new URL(request.url)
    const queryParams = Object.fromEntries(searchParams.entries())
    
    const validation = GetTaxonomySchema.safeParse(queryParams)
    if (!validation.success) {
      console.warn('Invalid query parameters', {
        errors: validation.error.issues,
        params: queryParams
      })
      return NextResponse.json(
        { 
          success: false, 
          error: 'Invalid query parameters',
          details: validation.error.issues
        },
        { status: 400 }
      )
    }

    const { parent_id, level, include_inactive, limit, offset } = validation.data

    console.info('Fetching taxonomy', { parent_id, level, include_inactive, limit, offset })

    // Fetch data based on parameters
    let categories: TaxonomyNode[]
    
    if (parent_id || level !== undefined) {
      // Get specific nodes
      categories = await DatabaseService.getTaxonomyNodes({
        parent_id,
        level,
        include_inactive,
        limit,
        offset
      })
    } else {
      // Get full tree
      categories = await DatabaseService.getTaxonomyTree({
        include_inactive,
        limit,
        offset
      })
    }

    console.info('Successfully fetched taxonomy', { count: categories.length })

    return NextResponse.json({
      success: true,
      data: categories,
      meta: {
        count: categories.length,
        limit,
        offset
      }
    } as ApiResponse<TaxonomyNode[]>)

  }, 'GET /api/taxonomy')
}

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
      const parent = await DatabaseService.getTaxonomyNode(parent_id)
      if (!parent) {
        console.warn('Parent taxonomy node not found', { parent_id })
        return NextResponse.json(
          { success: false, error: 'Parent category not found' },
          { status: 404 }
        )
      }
      level = parent.level + 1
      path = `${parent.path}${parent.id}/`
    }

    // Get next sort order
    const sort_order = await DatabaseService.getNextSortOrder(parent_id)

    // Create taxonomy node - แก้ไขชื่อ method ที่ถูกต้อง
    const category = await DatabaseService.createTaxonomyNode({
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
