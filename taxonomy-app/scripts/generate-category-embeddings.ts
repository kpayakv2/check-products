#!/usr/bin/env tsx
/**
 * Generate Embeddings for Taxonomy Categories
 * 
 * Uses local model (paraphrase-multilingual-MiniLM-L12-v2) via FastAPI
 * Same model as FastAPI backend for consistency (384-dim)
 */

import { createClient } from '@supabase/supabase-js'
import * as dotenv from 'dotenv'
import * as path from 'path'

// Load environment variables
dotenv.config({ path: path.join(__dirname, '../.env.local') })

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!
const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY!
const fastapiUrl = process.env.FASTAPI_URL || 'http://localhost:8000'

const supabase = createClient(supabaseUrl, supabaseKey)

interface Category {
  id: string
  name_th: string
  name_en: string | null
  description: string | null
  keywords: string[] | null
  embedding: number[] | null
}

async function generateEmbedding(text: string): Promise<number[]> {
  console.log(`  📝 Text: "${text.substring(0, 100)}..."`)
  
  const response = await fetch(`${fastapiUrl}/api/embed`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text })
  })

  if (!response.ok) {
    throw new Error(`FastAPI error: ${response.statusText}`)
  }

  const data = await response.json()
  return data.embedding
}

function generateCategoryText(category: Category): string {
  const parts: string[] = [category.name_th]
  
  if (category.name_en) {
    parts.push(category.name_en)
  }
  
  if (category.keywords && category.keywords.length > 0) {
    parts.push(...category.keywords)
  }
  
  if (category.description) {
    parts.push(category.description)
  }
  
  return parts.join(' ')
}

async function main() {
  console.log('🚀 Starting Category Embedding Generation')
  console.log('=' .repeat(60))
  console.log(`📡 Supabase: ${supabaseUrl}`)
  console.log(`🤖 FastAPI: ${fastapiUrl}`)
  console.log(`📊 Model: paraphrase-multilingual-MiniLM-L12-v2 (384-dim)`)
  console.log('=' .repeat(60))
  console.log()

  // 1. Check FastAPI is running
  console.log('1️⃣  Checking FastAPI connection...')
  try {
    const healthCheck = await fetch(`${fastapiUrl}/api/v1/health`)
    if (!healthCheck.ok) {
      throw new Error('FastAPI not responding')
    }
    const health = await healthCheck.json()
    console.log(`   ✅ FastAPI is running (v${health.version})`)
  } catch (error) {
    console.error('   ❌ FastAPI is not running!')
    console.error('   💡 Start FastAPI: python api_server.py')
    process.exit(1)
  }
  console.log()

  // 2. Load categories
  console.log('2️⃣  Loading taxonomy categories...')
  const { data: categories, error: loadError } = await supabase
    .from('taxonomy_nodes')
    .select('id, name_th, name_en, description, keywords, embedding')
    .eq('is_active', true)
    .order('level', { ascending: true })

  if (loadError) {
    console.error('   ❌ Failed to load categories:', loadError.message)
    process.exit(1)
  }

  console.log(`   ✅ Loaded ${categories.length} categories`)
  console.log()

  // 3. Filter categories that need embeddings
  const needsEmbedding = categories.filter(cat => !cat.embedding)
  const hasEmbedding = categories.filter(cat => cat.embedding)

  console.log(`   📊 Summary:`)
  console.log(`      - Has embeddings: ${hasEmbedding.length}`)
  console.log(`      - Needs embeddings: ${needsEmbedding.length}`)
  console.log()

  if (needsEmbedding.length === 0) {
    console.log('   ✅ All categories already have embeddings!')
    console.log()
    console.log('   💡 To regenerate all embeddings, run:')
    console.log('      UPDATE taxonomy_nodes SET embedding = NULL;')
    return
  }

  // 4. Confirm before proceeding
  console.log(`3️⃣  Ready to generate ${needsEmbedding.length} embeddings`)
  console.log(`   ⏱️  Estimated time: ${Math.ceil(needsEmbedding.length * 0.5)} seconds`)
  console.log()

  // 5. Generate embeddings
  console.log('4️⃣  Generating embeddings...')
  let successCount = 0
  let errorCount = 0

  for (let i = 0; i < needsEmbedding.length; i++) {
    const category = needsEmbedding[i]
    const progress = `[${i + 1}/${needsEmbedding.length}]`
    
    try {
      console.log(`\n${progress} ${category.name_th}`)
      
      // Generate text for embedding
      const text = generateCategoryText(category)
      
      // Generate embedding
      const embedding = await generateEmbedding(text)
      
      console.log(`  🎯 Embedding: ${embedding.length} dimensions`)
      
      // Save to database
      const { error: updateError } = await supabase
        .from('taxonomy_nodes')
        .update({ embedding: embedding })
        .eq('id', category.id)
      
      if (updateError) {
        throw updateError
      }
      
      console.log(`  ✅ Saved to database`)
      successCount++
      
      // Rate limiting (avoid overwhelming FastAPI)
      if (i < needsEmbedding.length - 1) {
        await new Promise(resolve => setTimeout(resolve, 100))
      }
      
    } catch (error) {
      console.error(`  ❌ Error:`, error)
      errorCount++
    }
  }

  // 6. Summary
  console.log()
  console.log('=' .repeat(60))
  console.log('✅ Embedding Generation Complete!')
  console.log('=' .repeat(60))
  console.log(`📊 Results:`)
  console.log(`   - Success: ${successCount}`)
  console.log(`   - Errors: ${errorCount}`)
  console.log(`   - Total: ${needsEmbedding.length}`)
  console.log()
  
  if (successCount > 0) {
    console.log('🎉 Category embeddings are ready!')
    console.log('💡 You can now use hybrid-classification-local Edge Function')
  }
  
  if (errorCount > 0) {
    console.log()
    console.log('⚠️  Some embeddings failed. Check the logs above.')
    console.log('   You can re-run this script to retry failed embeddings.')
  }
}

main().catch(error => {
  console.error('Fatal error:', error)
  process.exit(1)
})
