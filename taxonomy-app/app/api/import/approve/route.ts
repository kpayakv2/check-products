import { NextRequest, NextResponse } from 'next/server'
import { DatabaseService } from '@/utils/supabase'

export async function POST(request: NextRequest) {
  try {
    const { suggestions } = await request.json()

    if (!suggestions || !Array.isArray(suggestions)) {
      return NextResponse.json(
        { error: 'Invalid suggestions data' },
        { status: 400 }
      )
    }

    const results = {
      success: 0,
      failed: 0,
      errors: [] as string[]
    }

    // Create import batch record
    const importBatch = await DatabaseService.createImport({
      name: `Product Import - ${new Date().toISOString()}`,
      description: 'Bulk product import with AI category suggestions',
      total_records: suggestions.length,
      processed_records: 0,
      success_records: 0,
      error_records: 0,
      status: 'processing'
    })

    for (const suggestion of suggestions) {
      try {
        // Create product record
        const product = await DatabaseService.createProduct({
          name_th: suggestion.name_th,
          description: suggestion.cleaned_name,
          category_id: suggestion.suggested_category.id || null,
          keywords: suggestion.tokens,
          embedding: suggestion.embedding,
          metadata: {
            units: suggestion.units,
            attributes: suggestion.attributes,
            original_text: suggestion.name_th,
            cleaned_text: suggestion.cleaned_name
          },
          status: 'pending',
          confidence_score: suggestion.suggested_category.confidence_score,
          import_batch_id: importBatch.id
        })

        // Create category suggestion record
        if (suggestion.suggested_category.id) {
          await DatabaseService.createProductCategorySuggestion({
            product_id: product.id,
            suggested_category_id: suggestion.suggested_category.id,
            confidence_score: suggestion.suggested_category.confidence_score,
            suggestion_method: 'keyword_rule',
            metadata: {
              explanation: suggestion.suggested_category.explanation,
              matched_tokens: suggestion.tokens,
              processing_timestamp: new Date().toISOString()
            },
            is_accepted: true
          })
        }

        // Create product attributes
        if (suggestion.attributes) {
          for (const [key, value] of Object.entries(suggestion.attributes)) {
            if (Array.isArray(value)) {
              for (const item of value) {
                await DatabaseService.createProductAttribute({
                  product_id: product.id,
                  attribute_name: key,
                  attribute_value: String(item),
                  attribute_type: 'text'
                })
              }
            } else {
              await DatabaseService.createProductAttribute({
                product_id: product.id,
                attribute_name: key,
                attribute_value: String(value),
                attribute_type: typeof value === 'number' ? 'number' : 'text'
              })
            }
          }
        }

        results.success++
      } catch (error) {
        console.error(`Failed to create product: ${suggestion.name_th}`, error)
        results.failed++
        results.errors.push(`${suggestion.name_th}: ${error instanceof Error ? error.message : 'Unknown error'}`)
      }
    }

    // Update import batch status
    await DatabaseService.updateImport(importBatch.id, {
      processed_records: suggestions.length,
      success_records: results.success,
      error_records: results.failed,
      status: results.failed === 0 ? 'completed' : 'completed',
      error_details: results.errors.length > 0 ? { errors: results.errors } : null,
      completed_at: new Date().toISOString()
    })

    return NextResponse.json({
      success: true,
      results: {
        total: suggestions.length,
        success: results.success,
        failed: results.failed,
        import_batch_id: importBatch.id
      },
      errors: results.errors
    })

  } catch (error) {
    console.error('Approve suggestions error:', error)
    return NextResponse.json(
      { error: 'Failed to approve suggestions' },
      { status: 500 }
    )
  }
}
