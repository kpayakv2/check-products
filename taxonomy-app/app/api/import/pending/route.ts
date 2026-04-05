import { NextRequest, NextResponse } from 'next/server'
import { supabase } from '@/utils/supabase'

/**
 * GET /api/import/pending
 * ดึงรายการ suggestions ที่รอการ approve
 */
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url)
    const limit = parseInt(searchParams.get('limit') || '50')
    const offset = parseInt(searchParams.get('offset') || '0')
    const status = searchParams.get('status') || 'pending'

    // ดึง pending suggestions จาก product_category_suggestions
    const { data: suggestions, error, count } = await supabase
      .from('product_category_suggestions')
      .select(`
        *,
        suggested_category:taxonomy_nodes!suggested_category_id(
          id,
          name_th,
          code
        )
      `, { count: 'exact' })
      .is('is_accepted', null) // รอการ approve
      .eq('suggestion_method', 'hybrid_ai_preview') // จาก ProcessingStep
      .order('created_at', { ascending: false })
      .range(offset, offset + limit - 1)

    if (error) {
      console.error('Error fetching pending suggestions:', error)
      return NextResponse.json(
        { error: 'Failed to fetch pending suggestions' },
        { status: 500 }
      )
    }

    // Transform data สำหรับ UI
    const transformedSuggestions = suggestions?.map(suggestion => ({
      id: suggestion.id,
      product_name: suggestion.metadata?.product_name || 'Unknown',
      cleaned_name: suggestion.metadata?.cleaned_name || '',
      tokens: suggestion.metadata?.tokens || [],
      units: suggestion.metadata?.units || [],
      attributes: suggestion.metadata?.attributes || {},
      suggested_category: {
        id: suggestion.suggested_category?.id,
        name_th: suggestion.suggested_category?.name_th,
        code: suggestion.suggested_category?.code
      },
      confidence_score: suggestion.confidence_score,
      explanation: suggestion.metadata?.explanation || '',
      created_at: suggestion.created_at,
      status: 'pending'
    })) || []

    return NextResponse.json({
      success: true,
      data: transformedSuggestions,
      pagination: {
        total: count || 0,
        limit,
        offset,
        has_more: (count || 0) > offset + limit
      }
    })

  } catch (error) {
    console.error('Pending suggestions API error:', error)
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    )
  }
}

/**
 * POST /api/import/pending
 * Approve/Reject suggestions แบบ batch
 */
export async function POST(request: NextRequest) {
  try {
    const { action, suggestion_ids, batch_data } = await request.json()

    if (!action || !suggestion_ids || !Array.isArray(suggestion_ids)) {
      return NextResponse.json(
        { error: 'Invalid request data' },
        { status: 400 }
      )
    }

    const results = {
      success: 0,
      failed: 0,
      errors: [] as string[]
    }

    if (action === 'approve') {
      // Approve suggestions และสร้าง products
      for (const suggestionId of suggestion_ids) {
        try {
          // ดึงข้อมูล suggestion
          const { data: suggestion, error: fetchError } = await supabase
            .from('product_category_suggestions')
            .select('*')
            .eq('id', suggestionId)
            .single()

          if (fetchError || !suggestion) {
            throw new Error('Suggestion not found')
          }

          // สร้าง product record
          const { data: product, error: productError } = await supabase
            .from('products')
            .insert({
              name_th: suggestion.metadata?.product_name,
              description: suggestion.metadata?.cleaned_name,
              category_id: suggestion.suggested_category_id,
              keywords: suggestion.metadata?.tokens || [],
              metadata: {
                units: suggestion.metadata?.units || [],
                attributes: suggestion.metadata?.attributes || {},
                original_suggestion_id: suggestionId
              },
              status: 'approved',
              confidence_score: suggestion.confidence_score
            })
            .select()
            .single()

          if (productError) {
            throw new Error(`Failed to create product: ${productError.message}`)
          }

          // Update suggestion status
          const { error: updateError } = await supabase
            .from('product_category_suggestions')
            .update({
              is_accepted: true,
              reviewed_at: new Date().toISOString(),
              product_id: product.id
            })
            .eq('id', suggestionId)

          if (updateError) {
            throw new Error(`Failed to update suggestion: ${updateError.message}`)
          }

          // สร้าง product_attributes ถ้ามี
          const attributes = suggestion.metadata?.attributes || {}
          if (Object.keys(attributes).length > 0) {
            const attributeInserts = []
            
            for (const [attrName, attrValue] of Object.entries(attributes)) {
              if (attrValue && typeof attrValue === 'object' && 'matches' in attrValue) {
                const typedValue = attrValue as { matches: any[] }
                attributeInserts.push({
                  product_id: product.id,
                  attribute_name: attrName,
                  attribute_value: JSON.stringify(typedValue.matches),
                  attribute_type: 'regex_extracted'
                })
              }
            }

            if (attributeInserts.length > 0) {
              await supabase
                .from('product_attributes')
                .insert(attributeInserts)
            }
          }

          results.success++
        } catch (error) {
          console.error(`Failed to approve suggestion ${suggestionId}:`, error)
          results.failed++
          results.errors.push(`Suggestion ${suggestionId}: ${error instanceof Error ? error.message : 'Unknown error'}`)
        }
      }
    } else if (action === 'reject') {
      // Reject suggestions
      const { error: rejectError } = await supabase
        .from('product_category_suggestions')
        .update({
          is_accepted: false,
          reviewed_at: new Date().toISOString()
        })
        .in('id', suggestion_ids)

      if (rejectError) {
        return NextResponse.json(
          { error: 'Failed to reject suggestions' },
          { status: 500 }
        )
      }

      results.success = suggestion_ids.length
    }

    return NextResponse.json({
      success: true,
      action,
      results: {
        total: suggestion_ids.length,
        success: results.success,
        failed: results.failed
      },
      errors: results.errors
    })

  } catch (error) {
    console.error('Batch approval API error:', error)
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    )
  }
}
