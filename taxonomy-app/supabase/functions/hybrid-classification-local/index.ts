/// <reference types="https://deno.land/x/deno/lib/deno.ns.d.ts" />
import { serve } from 'https://deno.land/std@0.168.0/http/server.ts'
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
}

interface ClassificationRequest {
  product_name: string
  method?: 'hybrid' | 'keyword' | 'embedding'
  top_k?: number
  use_local_model?: boolean
}

interface CategorySuggestion {
  category_id: string
  category_name: string
  category_level: number
  confidence: number
  method: string
  matched_keyword?: string
  source?: string
  model?: string
}

serve(async (req) => {
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders })
  }

  try {
    const supabase = createClient(
      Deno.env.get('SUPABASE_URL') ?? '',
      Deno.env.get('SUPABASE_ANON_KEY') ?? '',
      {
        global: {
          headers: { Authorization: req.headers.get('Authorization')! },
        },
      }
    )

    const { 
      product_name, 
      method = 'hybrid', 
      top_k = 3,
      use_local_model = true 
    }: ClassificationRequest = await req.json()

    if (!product_name?.trim()) {
      return new Response(
        JSON.stringify({ error: 'Product name is required' }),
        { status: 400, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      )
    }

    const startTime = Date.now()
    let suggestions: CategorySuggestion[] = []

    if (method === 'hybrid') {
      console.log('🔄 Hybrid classification with database function...')
      
      // Generate embedding first
      const fastapiUrl = Deno.env.get('FASTAPI_URL') || 'http://host.docker.internal:8000'
      const embeddingResponse = await fetch(`${fastapiUrl}/api/embed`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: product_name })
      })

      if (!embeddingResponse.ok) {
        throw new Error('Failed to generate embedding from local model')
      }

      const embeddingData = await embeddingResponse.json()
      const embedding = embeddingData.embedding

      // Call database function for hybrid classification
      const { data: hybridResults, error: hybridError } = await supabase
        .rpc('hybrid_category_classification', {
          product_name: product_name,
          product_embedding: embedding,
          top_k: top_k
        })

      if (hybridError) {
        console.error('Hybrid classification error:', hybridError)
        throw hybridError
      }

      if (hybridResults) {
        console.log(`Found ${hybridResults.length} hybrid matches`)
        
        suggestions = hybridResults.map(match => ({
          category_id: match.category_id,
          category_name: match.category_name,
          category_level: match.category_level,
          confidence: match.confidence,
          method: match.method,
          matched_keyword: match.matched_keyword,
          source: 'database_function',
          model: 'paraphrase-multilingual-MiniLM-L12-v2'
        }))
      }

    } else if (method === 'embedding') {
      console.log('🧠 Embedding matching with local model...')
      
      const fastapiUrl = Deno.env.get('FASTAPI_URL') || 'http://host.docker.internal:8000'
      const embeddingResponse = await fetch(`${fastapiUrl}/api/embed`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: product_name })
      })

      if (!embeddingResponse.ok) {
        throw new Error('Failed to generate embedding from local model')
      }

      const embeddingData = await embeddingResponse.json()
      const embedding = embeddingData.embedding

      const { data: vectorMatches, error: vectorError } = await supabase
        .rpc('match_categories_by_embedding', {
          query_embedding: embedding,
          match_threshold: 0.3,
          match_count: top_k
        })

      if (vectorError) {
        console.error('Vector matching error:', vectorError)
        throw vectorError
      }

      if (vectorMatches) {
        console.log(`Found ${vectorMatches.length} vector matches`)
        
        suggestions = vectorMatches.map(match => ({
          category_id: match.category_id,
          category_name: match.category_name,
          category_level: match.category_level,
          confidence: match.similarity,
          method: 'embedding',
          source: 'pgvector',
          model: 'paraphrase-multilingual-MiniLM-L12-v2'
        }))
      }
    }

    // Sort by confidence and limit
    suggestions.sort((a, b) => b.confidence - a.confidence)
    suggestions = suggestions.slice(0, top_k)

    const processingTime = Date.now() - startTime

    return new Response(
      JSON.stringify({
        product_name,
        suggestions,
        top_suggestion: suggestions[0] || null,
        processing_time: processingTime,
        method,
        model: use_local_model ? 'local (paraphrase-multilingual-MiniLM-L12-v2)' : null,
        backend: 'Supabase Edge Function + FastAPI (local model)',
        algorithm: method === 'hybrid' ? 'Keyword 60% + Embedding 40%' : method,
        cost: 0, // FREE!
        note: 'Using same model and algorithm as FastAPI backend (72% accuracy)'
      }),
      {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        status: 200,
      }
    )

  } catch (error) {
    console.error('Classification error:', error)
    return new Response(
      JSON.stringify({ 
        error: error.message,
        help: 'Make sure FastAPI is running at http://localhost:8000 for local embeddings'
      }),
      {
        status: 500,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      }
    )
  }
})
