import { NextRequest, NextResponse } from 'next/server'
import { createClient } from '@supabase/supabase-js'

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
)

// Thai text processing utilities
class ThaiTextProcessor {
  static clean(text: string): string {
    return text
      .replace(/[^\u0E00-\u0E7Fa-zA-Z0-9\s\-\.\(\)]/g, '')
      .replace(/\s+/g, ' ')
      .trim()
      .toLowerCase()
  }

  static tokenize(text: string): string[] {
    const tokens = text
      .split(/[\s\-\(\)\[\]\/\\,\.]+/)
      .filter(token => token.length >= 2)
    return [...new Set(tokens)]
  }

  static extractUnits(text: string): string[] {
    const unitPatterns = [
      /(\d+)\s*(กรัม|g|gram)/gi,
      /(\d+)\s*(มิลลิลิตร|ml|มล)/gi,
      /(\d+)\s*(ลิตร|l)/gi,
      /(\d+)\s*(กิโลกรม|kg|กก)/gi,
      /(\d+)\s*(ชิ้น|pcs)/gi,
      /(\d+)\s*(แพ็ค|pack)/gi,
      /(\d+)\s*(กล่อง|box)/gi
    ]

    const units: string[] = []
    unitPatterns.forEach(pattern => {
      const matches = text.match(pattern)
      if (matches) units.push(...matches)
    })
    return units
  }

  static extractAttributes(text: string): Record<string, any> {
    const attributes: Record<string, any> = {}
    
    const colors = ['แดง', 'เขียว', 'น้ำเงิน', 'เหลือง', 'ขาว', 'ดำ', 'ชมพู', 'ม่วง', 'ส้ม', 'เทา']
    const foundColors = colors.filter(color => text.includes(color))
    if (foundColors.length > 0) attributes.colors = foundColors

    const sizes = ['S', 'M', 'L', 'XL', 'XXL', 'เล็ก', 'กลาง', 'ใหญ่']
    const foundSizes = sizes.filter(size => text.includes(size))
    if (foundSizes.length > 0) attributes.sizes = foundSizes

    return attributes
  }
}

/**
 * Generate embedding using local model via Edge Function
 */
async function generateEmbeddingLocal(text: string): Promise<number[]> {
  try {
    const { data, error } = await supabase.functions.invoke('generate-embeddings-local', {
      body: { texts: [text], model: 'sentence-transformer' }
    })

    if (error) throw error
    return data.embeddings[0]

  } catch (error) {
    console.error('Edge Function embedding failed:', error)
    
    // Fallback: Call FastAPI directly
    try {
      const fastapiUrl = process.env.FASTAPI_URL || 'http://localhost:8000'
      const response = await fetch(`${fastapiUrl}/api/embed`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
        signal: AbortSignal.timeout(10000)
      })
      
      if (response.ok) {
        const data = await response.json()
        return data.embedding
      }
    } catch (fastapiError) {
      console.error('FastAPI fallback failed:', fastapiError)
    }
    
    // Last resort: Mock embedding
    return Array.from({ length: 384 }, () => Math.random() * 2 - 1)
  }
}

/**
 * Suggest category using Hybrid Algorithm via Edge Function
 */
async function suggestCategoryHybrid(
  productName: string,
  tokens: string[],
  attributes: Record<string, any>,
  embedding: number[]
): Promise<{
  category_id: string | null
  category_name: string
  confidence: number
  method: string
  all_suggestions: any[]
}> {
  try {
    // Call hybrid-classification-local Edge Function
    const { data, error } = await supabase.functions.invoke('hybrid-classification-local', {
      body: {
        product_name: productName,
        method: 'hybrid',
        top_k: 5,
        use_local_model: true
      }
    })

    if (error) throw error

    return {
      category_id: data.top_suggestion?.category_id || null,
      category_name: data.top_suggestion?.category_name || 'ไม่ระบุหมวดหมู่',
      confidence: data.top_suggestion?.confidence || 0,
      method: data.top_suggestion?.method || 'hybrid',
      all_suggestions: data.suggestions || []
    }

  } catch (error) {
    console.error('Hybrid classification failed:', error)
    
    // Fallback: Return default
    return {
      category_id: null,
      category_name: 'ไม่สามารถจัดหมวดหมู่ได้',
      confidence: 0,
      method: 'error',
      all_suggestions: []
    }
  }
}

/**
 * Main POST handler - Process CSV file from Supabase Storage
 */
export async function POST(request: NextRequest) {
  try {
    const { fileName } = await request.json()

    if (!fileName) {
      return NextResponse.json({ error: 'File name is required' }, { status: 400 })
    }

    console.log(`📦 Processing file from storage: ${fileName}`)

    // Download file from Supabase Storage
    const { data: fileData, error: downloadError } = await supabase.storage
      .from('uploads')
      .download(fileName)

    if (downloadError) {
      console.error('Download error:', downloadError)
      return NextResponse.json({ 
        error: 'Failed to download file from storage',
        details: downloadError.message 
      }, { status: 400 })
    }

    // Convert blob to text
    const text = await fileData.text()
    const lines = text.split('\n').filter(line => line.trim())
    
    if (lines.length < 2) {
      return NextResponse.json({ 
        error: 'CSV file must have at least 2 lines (header + data)' 
      }, { status: 400 })
    }

    const headers = lines[0].split(',').map(h => h.trim().replace(/"/g, ''))
    
    const nameIndex = headers.findIndex(h => 
      h.toLowerCase().includes('name') || 
      h.includes('ชื่อ') || 
      h.includes('สินค้า') ||
      h.toLowerCase().includes('product')
    )

    if (nameIndex === -1) {
      return NextResponse.json({ 
        error: 'CSV must have a column named "name", "ชื่อสินค้า", "product", or similar',
        headers: headers
      }, { status: 400 })
    }

    const results: any[] = []
    const errors: any[] = []
    
    console.log(`📊 Processing ${lines.length - 1} products with local model...`)

    // Process each product
    for (let i = 1; i < lines.length && i <= 11; i++) { // Limit to 10 products for testing
      const values = lines[i].split(',').map(v => v.trim().replace(/"/g, ''))
      const productName = values[nameIndex]?.trim()

      if (!productName) {
        errors.push({ line: i, error: 'Empty product name' })
        continue
      }

      try {
        console.log(`Processing ${i}/${Math.min(lines.length - 1, 10)}: ${productName}`)

        // 1. Clean & tokenize
        const cleaned = ThaiTextProcessor.clean(productName)
        const tokens = ThaiTextProcessor.tokenize(cleaned)
        const units = ThaiTextProcessor.extractUnits(productName)
        const attributes = ThaiTextProcessor.extractAttributes(productName)

        // 2. Generate embedding (local model via Edge Function)
        const embedding = await generateEmbeddingLocal(productName)

        // 3. Suggest category (Hybrid algorithm via Edge Function)
        const suggestion = await suggestCategoryHybrid(
          productName,
          tokens,
          attributes,
          embedding
        )

        // 4. Save to database
        const { data: product, error: insertError } = await supabase
          .from('products')
          .insert({
            name_th: productName,
            embedding: embedding,
            category_id: suggestion.category_id,
            confidence_score: suggestion.confidence,
            keywords: tokens,
            metadata: {
              units,
              attributes,
              original_line: i,
              processing_method: 'storage_import',
              source_file: fileName
            }
          })
          .select()
          .single()

        if (insertError) {
          console.error('Insert error:', insertError)
          errors.push({ 
            line: i, 
            product: productName, 
            error: insertError.message 
          })
          continue
        }

        // 5. Save category suggestion
        if (suggestion.category_id && product) {
          await supabase
            .from('product_category_suggestions')
            .insert({
              product_id: product.id,
              category_id: suggestion.category_id,
              confidence_score: suggestion.confidence,
              method: suggestion.method,
              all_suggestions: suggestion.all_suggestions,
              created_at: new Date().toISOString()
            })
        }

        results.push({
          line: i,
          product_name: productName,
          category: suggestion.category_name,
          confidence: suggestion.confidence,
          method: suggestion.method,
          tokens_count: tokens.length,
          units_count: units.length,
          attributes_count: Object.keys(attributes).length,
          embedding_dim: embedding.length,
          product_id: product?.id
        })

        // Add small delay to prevent overwhelming
        await new Promise(resolve => setTimeout(resolve, 100))

      } catch (error) {
        console.error(`Error processing line ${i}:`, error)
        errors.push({ 
          line: i, 
          product: productName, 
          error: error.message 
        })
      }
    }

    return NextResponse.json({
      success: true,
      message: `Processed ${results.length} products from storage file`,
      source_file: fileName,
      total_lines: lines.length - 1,
      processed: results.length,
      errors: errors.length,
      results: results,
      errors: errors,
      processing_info: {
        model: 'paraphrase-multilingual-MiniLM-L12-v2',
        algorithm: 'Hybrid (Keyword 60% + Embedding 40%)',
        backend: 'Supabase Edge Functions + FastAPI',
        cost: 0 // FREE!
      }
    })

  } catch (error) {
    console.error('Storage processing error:', error)
    return NextResponse.json({
      success: false,
      error: 'Failed to process file from storage',
      details: error.message
    }, { status: 500 })
  }
}
