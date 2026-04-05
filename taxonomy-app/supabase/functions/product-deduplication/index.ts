/// <reference types="https://deno.land/x/deno/lib/deno.ns.d.ts" />
import { serve } from "https://deno.land/std@0.168.0/http/server.ts"
import { createClient } from '@supabase/supabase-js'

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
}

interface DeduplicationRequest {
  oldProductsPath: string
  newProductsPath: string
  threshold?: number
  method?: 'tfidf' | 'semantic'
}

interface ProductMatch {
  id: string
  newProduct: string
  oldProduct: string
  similarity: number
  confidence: number
  mlPrediction: 'similar' | 'different'
  status: 'pending'
  reason: string
}

serve(async (req) => {
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders })
  }

  try {
    // Validate environment variables
    const supabaseUrl = Deno.env.get('SUPABASE_URL')
    const supabaseKey = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY')
    
    if (!supabaseUrl || !supabaseKey) {
      throw new Error('Missing Supabase configuration')
    }

    const supabase = createClient(supabaseUrl, supabaseKey)

    const { 
      oldProductsPath, 
      newProductsPath, 
      threshold = 0.75,
      method = 'tfidf' 
    }: DeduplicationRequest = await req.json()

    if (!oldProductsPath || !newProductsPath) {
      return new Response(
        JSON.stringify({ error: 'Missing file paths' }),
        { status: 400, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      )
    }

    const startTime = Date.now()

    // Download files from Supabase Storage
    const { data: oldFileData } = await supabase.storage
      .from('uploads')
      .download(oldProductsPath)

    const { data: newFileData } = await supabase.storage
      .from('uploads')
      .download(newProductsPath)

    if (!oldFileData || !newFileData) {
      throw new Error('Failed to download files')
    }

    // Parse CSV data
    const oldProducts = await parseCSV(oldFileData)
    const newProducts = await parseCSV(newFileData)

    console.log(`Processing ${newProducts.length} new products vs ${oldProducts.length} old products`)

    // Perform deduplication analysis
    const matches: ProductMatch[] = []
    const uniqueProducts: string[] = []

    for (let i = 0; i < newProducts.length; i++) {
      const newProduct = newProducts[i]
      let bestMatch = { product: '', similarity: 0 }

      // Find best match in old products
      for (const oldProduct of oldProducts) {
        const similarity = calculateSimilarity(newProduct, oldProduct, method)
        if (similarity > bestMatch.similarity) {
          bestMatch = { product: oldProduct, similarity }
        }
      }

      // Classify products based on similarity
      if (bestMatch.similarity >= 0.95) {
        // Very high similarity - likely duplicate, auto-exclude
        continue
      } else if (bestMatch.similarity >= threshold) {
        // Medium similarity - needs human review
        const confidence = Math.min(0.95, bestMatch.similarity + 0.1)
        const mlPrediction = bestMatch.similarity > 0.8 ? 'similar' : 'different'

        matches.push({
          id: `review_${i + 1}`,
          newProduct: newProduct,
          oldProduct: bestMatch.product,
          similarity: bestMatch.similarity,
          confidence: confidence,
          mlPrediction: mlPrediction,
          status: 'pending',
          reason: `ความคล้าย ${(bestMatch.similarity * 100).toFixed(1)}% - ${
            mlPrediction === 'similar' ? 'อาจซ้ำกัน' : 'อาจแตกต่าง'
          }`
        })
      } else {
        // Low similarity - likely unique
        uniqueProducts.push(newProduct)
      }
    }

    const processingTime = Date.now() - startTime

    // Calculate statistics
    const excludedDuplicates = newProducts.length - uniqueProducts.length - matches.length
    const stats = {
      totalNewProducts: newProducts.length,
      totalOldProducts: oldProducts.length,
      needsReview: matches.length,
      autoApproved: uniqueProducts.length,
      excludedDuplicates: excludedDuplicates,
      processingTime: processingTime
    }

    return new Response(
      JSON.stringify({
        success: true,
        results: matches,
        uniqueProducts: uniqueProducts,
        stats: stats,
        summary: `จาก ${newProducts.length} สินค้าใหม่: ${uniqueProducts.length} ไม่ซ้ำ, ${matches.length} ต้องตรวจสอบ, ${excludedDuplicates} ซ้ำมาก (ตัดออก)`
      }),
      {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      }
    )

  } catch (error) {
    console.error('Deduplication error:', error)
    const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred'
    return new Response(
      JSON.stringify({ error: errorMessage }),
      {
        status: 500,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      }
    )
  }
})

// Helper functions
async function parseCSV(fileData: Blob): Promise<string[]> {
  const text = await fileData.text()
  const lines = text.split('\n').filter(line => line.trim())
  
  // Skip header and extract product names
  return lines.slice(1).map(line => {
    const columns = line.split(',')
    // Try different column patterns for product names
    return columns[0]?.replace(/"/g, '').trim() || 
           columns[1]?.replace(/"/g, '').trim() || 
           line.trim()
  }).filter(name => name.length > 0)
}

function calculateSimilarity(text1: string, text2: string, method: string): number {
  if (method === 'semantic') {
    // TODO: Implement semantic similarity using embeddings
    return calculateTFIDFSimilarity(text1, text2)
  }
  
  return calculateTFIDFSimilarity(text1, text2)
}

function calculateTFIDFSimilarity(text1: string, text2: string): number {
  // Simple TF-IDF-like similarity calculation
  const normalize = (text: string) => text.toLowerCase()
    .replace(/[^\u0E00-\u0E7Fa-zA-Z0-9\s]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()

  const tokens1 = normalize(text1).split(' ')
  const tokens2 = normalize(text2).split(' ')

  if (tokens1.length === 0 || tokens2.length === 0) return 0

  // Calculate Jaccard similarity
  const set1 = new Set(tokens1)
  const set2 = new Set(tokens2)
  const intersection = new Set([...set1].filter(x => set2.has(x)))
  const union = new Set([...set1, ...set2])

  const jaccard = intersection.size / union.size

  // Calculate token overlap ratio
  const commonTokens = tokens1.filter(token => tokens2.includes(token))
  const overlapRatio = commonTokens.length / Math.max(tokens1.length, tokens2.length)

  // Combine metrics
  return (jaccard * 0.6) + (overlapRatio * 0.4)
}
