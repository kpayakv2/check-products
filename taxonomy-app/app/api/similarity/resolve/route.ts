import { NextRequest, NextResponse } from 'next/server'
import { z } from 'zod'
import { supabaseAdmin } from '@/utils/supabase-admin'
import { rateLimit } from '@/utils/rate-limit'
import { withErrorHandling } from '@/utils/error-handler'
import { validateRequest } from '@/utils/validation'

// Validation Schema
const ResolveDuplicateSchema = z.object({
  action: z.enum(['merge', 'keep_both']),
  id_a: z.string().uuid('id_a ต้องเป็น UUID ที่ถูกต้อง'),
  id_b: z.string().uuid('id_b ต้องเป็น UUID ที่ถูกต้อง'),
  keep_id: z.string().uuid('keep_id ต้องเป็น UUID ที่ถูกต้อง').optional(),
  similarity: z.coerce.number().min(0).max(1).default(1.0)
})

const limiter = rateLimit({
  interval: 60 * 1000, // 1 minute
  uniqueTokenPerInterval: 500, // Max 500 unique IPs per minute
})

/**
 * POST /api/similarity/resolve
 * จัดการสะสางสินค้าที่ซ้ำซ้อนภายในคลังสินค้า (Merge/Keep Both)
 */
export async function POST(request: NextRequest): Promise<NextResponse> {
  return withErrorHandling(async () => {
    // Rate limiting
    try {
      await limiter.check(20, 'similarity-resolve') // 20 requests per minute per IP
    } catch {
      return NextResponse.json(
        { success: false, error: 'Too many requests' },
        { status: 429 }
      )
    }

    // Parse and validate request body
    const body = await validateRequest(request, ResolveDuplicateSchema)
    if ('error' in body) {
      return NextResponse.json(body, { status: 400 })
    }

    const { action, id_a, id_b, keep_id, similarity } = body

    console.info('Resolving duplicate pair', { action, id_a, id_b, keep_id })

    // Sort IDs to ensure consistent unique constraint in similarity_matches (product_a_id < product_b_id)
    const [prod_a_id, prod_b_id] = [id_a, id_b].sort()

    if (action === 'merge') {
      if (!keep_id) {
        return NextResponse.json(
          { success: false, error: 'keep_id is required for merge action' },
          { status: 400 }
        )
      }

      const remove_id = keep_id === id_a ? id_b : id_a

      // Soft delete: Update the duplicate product status to 'duplicate'
      const { error: updateError } = await supabaseAdmin
        .from('products')
        .update({ status: 'duplicate' })
        .eq('id', remove_id)

      if (updateError) {
        console.error('Failed to update duplicate product status', updateError)
        return NextResponse.json(
          { success: false, error: 'Failed to update product status: ' + updateError.message },
          { status: 500 }
        )
      }

      // Record match decision in similarity_matches table
      const { error: matchError } = await supabaseAdmin
        .from('similarity_matches')
        .upsert({
          product_a_id: prod_a_id,
          product_b_id: prod_b_id,
          similarity_score: similarity,
          match_type: 'semantic',
          algorithm: 'cosine',
          is_duplicate: true,
          reviewed: true,
          reviewed_at: new Date().toISOString(),
          metadata: {
            resolved_action: 'merge',
            keep_product_id: keep_id,
            removed_product_id: remove_id,
            resolved_at: new Date().toISOString()
          }
        }, { onConflict: 'product_a_id, product_b_id' })

      if (matchError) {
        console.error('Failed to insert similarity match record', matchError)
        return NextResponse.json(
          { success: false, error: 'Failed to record match decision: ' + matchError.message },
          { status: 500 }
        )
      }

      return NextResponse.json({
        success: true,
        message: 'Products successfully merged. Duplicate product set to duplicate status.'
      })

    } else if (action === 'keep_both') {
      // Keep both products as approved.
      // Record match decision in similarity_matches as NOT duplicates
      const { error: matchError } = await supabaseAdmin
        .from('similarity_matches')
        .upsert({
          product_a_id: prod_a_id,
          product_b_id: prod_b_id,
          similarity_score: similarity,
          match_type: 'semantic',
          algorithm: 'cosine',
          is_duplicate: false,
          reviewed: true,
          reviewed_at: new Date().toISOString(),
          metadata: {
            resolved_action: 'keep_both',
            resolved_at: new Date().toISOString()
          }
        }, { onConflict: 'product_a_id, product_b_id' })

      if (matchError) {
        console.error('Failed to insert similarity match record', matchError)
        return NextResponse.json(
          { success: false, error: 'Failed to record match decision: ' + matchError.message },
          { status: 500 }
        )
      }

      return NextResponse.json({
        success: true,
        message: 'Decided to keep both products. Recorded exception in similarity matches.'
      })
    }

    return NextResponse.json(
      { success: false, error: 'Invalid action' },
      { status: 400 }
    )
  }, 'POST /api/similarity/resolve')
}
