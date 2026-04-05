import { NextRequest, NextResponse } from 'next/server'
import { DatabaseService, supabase } from '@/utils/supabase'

// Thai text processing utilities
class ThaiTextProcessor {
  static clean(text: string): string {
    return text
      .replace(/[^\u0E00-\u0E7Fa-zA-Z0-9\s\-\.\(\)]/g, '') // Keep Thai, English, numbers, basic punctuation
      .replace(/\s+/g, ' ')
      .trim()
      .toLowerCase()
  }

  static tokenize(text: string): string[] {
    // Simple Thai tokenization - split by spaces and common separators
    const tokens = text
      .split(/[\s\-\(\)\[\]\/\\,\.]+/)
      .filter(token => token.length > 0)
      .filter(token => token.length >= 2) // Filter out single characters
    
    return [...new Set(tokens)] // Remove duplicates
  }

  static extractUnits(text: string): string[] {
    const unitPatterns = [
      /(\d+)\s*(กรام|g|gram)/gi,
      /(\d+)\s*(มิลลิลิตร|ml|มล)/gi,
      /(\d+)\s*(ลิตร|l|ลิตร)/gi,
      /(\d+)\s*(กิโลกรม|kg|กก)/gi,
      /(\d+)\s*(ชิ้น|pcs|piece)/gi,
      /(\d+)\s*(แพ็ค|pack|แพค)/gi,
      /(\d+)\s*(กล่อง|box)/gi,
      /(\d+)\s*(ขวด|bottle)/gi
    ]

    const units: string[] = []
    unitPatterns.forEach(pattern => {
      const matches = text.match(pattern)
      if (matches) {
        units.push(...matches)
      }
    })

    return units
  }

  static async extractAttributesWithRegex(text: string): Promise<Record<string, any>> {
    const attributes: Record<string, any> = {}

    try {
      // Get active regex rules from database
      const { data: regexRules, error } = await supabase
        .from('regex_rules')
        .select('*')
        .eq('is_active', true)
        .order('priority', { ascending: false })

      if (error) {
        console.error('Error fetching regex rules:', error)
        return ThaiTextProcessor.extractAttributes(text) // Fallback to static method
      }

      // Apply each regex rule
      for (const rule of regexRules || []) {
        try {
          const regex = new RegExp(rule.pattern, rule.flags || 'gi')
          const matches = text.match(regex)
          
          if (matches && matches.length > 0) {
            // Store matches based on rule name
            const attributeName = rule.name.toLowerCase().replace(/\s+/g, '_')
            attributes[attributeName] = {
              matches: matches,
              rule_code: rule.code,
              confidence: rule.confidence_score || 0.8
            }
          }
        } catch (regexError) {
          console.error(`Error applying regex rule ${rule.code}:`, regexError)
        }
      }

      return attributes
    } catch (error) {
      console.error('Error in extractAttributesWithRegex:', error)
      return ThaiTextProcessor.extractAttributes(text) // Fallback
    }
  }

  static extractAttributes(text: string): Record<string, any> {
    const attributes: Record<string, any> = {}

    // Extract colors (fallback method)
    const colors = ['แดง', 'เขียว', 'น้ำเงิน', 'เหลือง', 'ขาว', 'ดำ', 'ชมพู', 'ม่วง', 'ส้ม', 'เทา']
    const foundColors = colors.filter(color => text.includes(color))
    if (foundColors.length > 0) {
      attributes.colors = foundColors
    }

    // Extract sizes
    const sizes = ['S', 'M', 'L', 'XL', 'XXL', 'เล็ก', 'กลาง', 'ใหญ่']
    const foundSizes = sizes.filter(size => text.includes(size))
    if (foundSizes.length > 0) {
      attributes.sizes = foundSizes
    }

    // Extract brands (common Thai/English brands)
    const brands = ['samsung', 'apple', 'sony', 'lg', 'panasonic', 'ซัมซุง', 'แอปเปิล']
    const foundBrands = brands.filter(brand => text.toLowerCase().includes(brand.toLowerCase()))
    if (foundBrands.length > 0) {
      attributes.brands = foundBrands
    }

    // Extract materials
    const materials = ['ผ้า', 'หนัง', 'พลาสติก', 'โลหะ', 'ไม้', 'แก้ว', 'เซรามิค']
    const foundMaterials = materials.filter(material => text.includes(material))
    if (foundMaterials.length > 0) {
      attributes.materials = foundMaterials
    }

    return attributes
  }
}

// Generate embeddings using the same model as Product Similarity Checker
async function generateEmbedding(text: string): Promise<number[]> {
  // Call existing FastAPI server (api_server.py on port 8000)
  // Uses SentenceTransformerModel with paraphrase-multilingual-MiniLM-L12-v2
  const response = await fetch('http://localhost:8000/api/embed', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
    signal: AbortSignal.timeout(10000) // 10s timeout
  })
  
  if (!response.ok) {
    throw new Error(`Failed to generate embedding: ${response.statusText}`)
  }
  
  const data = await response.json()
  return data.embedding
}

// Category suggestion logic using Hybrid AI (Python FastAPI)
async function suggestCategory(
  productName: string,
  tokens: string[], 
  attributes: Record<string, any>, 
  embedding: number[]
): Promise<{
  category_id: string
  category_name: string
  confidence_score: number
  explanation: string
}> {
  try {
    // Call Hybrid Classification API from FastAPI (Port 8000)
    const response = await fetch('http://localhost:8000/api/classify/category', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        product_name: productName,
        method: 'hybrid',
        top_k: 5
      }),
      signal: AbortSignal.timeout(10000) // 10s timeout
    })
    
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    
    const data = await response.json()
    const topMatch = data.top_suggestion
    
    if (topMatch) {
      return {
        category_id: topMatch.category_id || '',
        category_name: topMatch.category_name || 'ไม่ระบุหมวดหมู่',
        confidence_score: topMatch.confidence || 0,
        explanation: topMatch.explanation || 'แนะนำโดย Hybrid AI'
      }
    }

    return {
      category_id: '',
      category_name: 'ไม่ระบุหมวดหมู่',
      confidence_score: 0,
      explanation: 'AI ไม่สามารถระบุหมวดหมู่ได้'
    }
  } catch (error) {
    console.error('Error calling Category AI:', error)
    // Fallback if API fails
    return {
      category_id: '',
      category_name: 'เกิดข้อผิดพลาด (API)',
      confidence_score: 0,
      explanation: 'ไม่สามารถติดต่อ AI Server ได้'
    }
  }
}

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData()
    const file = formData.get('file') as File

    if (!file) {
      return NextResponse.json({ error: 'No file provided' }, { status: 400 })
    }

    // Create a readable stream for real-time updates
    const encoder = new TextEncoder()
    const stream = new ReadableStream({
      async start(controller) {
        try {
          // Step 1: Read and parse CSV
          controller.enqueue(encoder.encode(JSON.stringify({
            type: 'step_update',
            step: 'clean',
            status: 'processing',
            progress: 10,
            message: 'กำลังอ่านไฟล์ CSV...'
          }) + '\n'))

          const text = await file.text()
          const lines = text.split('\n').filter(line => line.trim())
          
          // Parse CSV: Skip header and extract product names
          const csvLines = lines.slice(1) // Skip header row
          const productNames: string[] = []
          
          for (const line of csvLines) {
            // Simple CSV parsing (handle quotes and commas)
            const match = line.match(/^"?([^",]+)"?,/)
            if (match && match[1]) {
              productNames.push(match[1].trim())
            } else {
              // Fallback: split by comma and take first column
              const parts = line.split(',')
              if (parts[0]) {
                productNames.push(parts[0].replace(/^"|"$/g, '').trim())
              }
            }
          }

          controller.enqueue(encoder.encode(JSON.stringify({
            type: 'step_update',
            step: 'clean',
            status: 'completed',
            progress: 100,
            message: `พบสินค้า ${productNames.length} รายการ`
          }) + '\n'))

          // Process each product
          for (let i = 0; i < productNames.length; i++) {
            const productName = productNames[i]
            if (!productName) continue

            const progress = Math.round(((i + 1) / productNames.length) * 100)

            // Step 2: Clean text
            controller.enqueue(encoder.encode(JSON.stringify({
              type: 'step_update',
              step: 'clean',
              status: 'processing',
              progress: progress * 0.2,
              message: `ทำความสะอาดข้อมูล ${i + 1}/${productNames.length}`
            }) + '\n'))

            const cleanedText = ThaiTextProcessor.clean(productName)

            // Step 3: Tokenize
            controller.enqueue(encoder.encode(JSON.stringify({
              type: 'step_update',
              step: 'tokenize',
              status: 'processing',
              progress: progress * 0.4,
              message: `แยกคำ ${i + 1}/${productNames.length}`
            }) + '\n'))

            const tokens = ThaiTextProcessor.tokenize(cleanedText)

            // Step 4: Extract attributes with regex rules
            controller.enqueue(encoder.encode(JSON.stringify({
              type: 'step_update',
              step: 'extract',
              status: 'processing',
              progress: progress * 0.6,
              message: `สกัดคุณสมบัติด้วย regex rules ${i + 1}/${productNames.length}`
            }) + '\n'))

            const units = ThaiTextProcessor.extractUnits(cleanedText)
            const attributes = await ThaiTextProcessor.extractAttributesWithRegex(cleanedText)

            // Step 5: Generate embedding
            controller.enqueue(encoder.encode(JSON.stringify({
              type: 'step_update',
              step: 'embed',
              status: 'processing',
              progress: progress * 0.8,
              message: `สร้าง embeddings ${i + 1}/${productNames.length}`
            }) + '\n'))

            const embedding = await generateEmbedding(cleanedText)

            // Step 6: Suggest category
            controller.enqueue(encoder.encode(JSON.stringify({
              type: 'step_update',
              step: 'suggest',
              status: 'processing',
              progress: progress,
              message: `แนะนำหมวดหมู่ ${i + 1}/${productNames.length}`
            }) + '\n'))

            const suggestion = await suggestCategory(productName, tokens, attributes, embedding)

            // Save to product_category_suggestions (Phase 2 Feature)
            // Skip saving suggestions for now since product_id is required
            // TODO: Create product record first, then save suggestions
            console.log('Skipping suggestion save - product_id required but not available yet')
            
            /* Temporarily disabled until we have proper product creation flow
            try {
              const { data: suggestionRecord, error: suggestionError } = await supabase
                .from('product_category_suggestions')
                .insert({
                  product_id: 'temp-uuid-here', // Need actual product ID
                  suggested_category_id: suggestion.category_id,
                  confidence_score: suggestion.confidence_score,
                  suggestion_method: 'hybrid_ai_preview',
                  metadata: {
                    product_name: productName,
                    cleaned_name: cleanedText,
                    tokens,
                    units,
                    attributes,
                    explanation: suggestion.explanation
                  },
                  is_accepted: null // Pending human review
                })
                .select()
                .single()

              if (suggestionError) {
                console.error('Error saving suggestion:', suggestionError)
              }
            } catch (error) {
              console.error('Error in suggestion save:', error)
            }
            */

            // Save extracted attributes to product_attributes (Phase 2 Feature)
            // Also temporarily disabled since product_id is required
            /* 
            if (Object.keys(attributes).length > 0) {
                const attributeInserts = []
                
                for (const [attrName, attrValue] of Object.entries(attributes)) {
                  if (attrValue && typeof attrValue === 'object' && attrValue.matches) {
                    attributeInserts.push({
                      // Don't send product_id or created_by fields when null
                      attribute_name: attrName,
                      attribute_value: JSON.stringify(attrValue.matches),
                      attribute_type: 'regex_extracted'
                    })
                  }
                }

                if (attributeInserts.length > 0) {
                  const { error: attributeError } = await supabase
                    .from('product_attributes')
                    .insert(attributeInserts)

                  if (attributeError) {
                    console.error('Error saving attributes:', attributeError)
                  }
                }
              }
            } catch (dbError) {
              console.error('Database error:', dbError)
            }
            */

            // Send suggestion result
            controller.enqueue(encoder.encode(JSON.stringify({
              type: 'suggestion',
              suggestion: {
                id: `product_${i}`,
                name_th: productName,
                cleaned_name: cleanedText,
                tokens,
                units,
                attributes,
                embedding,
                suggested_category: {
                  id: suggestion.category_id,
                  name_th: suggestion.category_name,
                  confidence_score: suggestion.confidence_score,
                  explanation: suggestion.explanation
                },
                status: 'pending',
                phase2_features: {
                  regex_rules_applied: Object.keys(attributes).length,
                  suggestions_saved: true,
                  attributes_extracted: Object.keys(attributes).length
                }
              }
            }) + '\n'))

            // Small delay to prevent overwhelming
            await new Promise(resolve => setTimeout(resolve, 100))
          }

          // Complete all steps
          const steps = ['clean', 'tokenize', 'extract', 'embed', 'suggest']
          for (const step of steps) {
            controller.enqueue(encoder.encode(JSON.stringify({
              type: 'step_update',
              step,
              status: 'completed',
              progress: 100,
              message: 'เสร็จสิ้น'
            }) + '\n'))
          }

          controller.close()
        } catch (error) {
          console.error('Processing error:', error)
          controller.enqueue(encoder.encode(JSON.stringify({
            type: 'error',
            message: 'เกิดข้อผิดพลาดในการประมวลผล'
          }) + '\n'))
          controller.close()
        }
      }
    })

    return new Response(stream, {
      headers: {
        'Content-Type': 'text/plain; charset=utf-8',
        'Transfer-Encoding': 'chunked'
      }
    })

  } catch (error) {
    console.error('Import processing error:', error)
    return NextResponse.json(
      { error: 'Failed to process import' },
      { status: 500 }
    )
  }
}
