/// <reference types="https://deno.land/x/deno/lib/deno.ns.d.ts" />
import { serve } from "https://deno.land/std@0.168.0/http/server.ts"
import { createClient } from '@supabase/supabase-js'

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
}

interface SearchRequest {
  query: string
  type: 'vector' | 'text' | 'hybrid'
  filters?: {
    categories?: string[]
    priceRange?: [number, number]
    confidence?: [number, number]
  }
  limit?: number
  offset?: number
}

serve(async (req) => {
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders })
  }

  try {
    // Validate environment variables
    const supabaseUrl = Deno.env.get('SUPABASE_URL')
    const supabaseKey = Deno.env.get('SUPABASE_ANON_KEY')
    
    if (!supabaseUrl || !supabaseKey) {
      throw new Error('Missing Supabase configuration')
    }

    const supabase = createClient(
      supabaseUrl,
      supabaseKey,
      {
        global: {
          headers: { Authorization: req.headers.get('Authorization')! },
        },
      }
    )

    const { query, type, filters, limit = 50, offset = 0 }: SearchRequest = await req.json()

    if (!query?.trim()) {
      return new Response(
        JSON.stringify({ error: 'Query is required' }),
        { status: 400, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      )
    }

    const startTime = Date.now()
    let results: any[] = []

    // Generate embedding for vector search
    let queryEmbedding: number[] = []
    if (type === 'vector' || type === 'hybrid') {
      const openaiKey = Deno.env.get('OPENAI_API_KEY')
      if (!openaiKey) {
        console.warn('OpenAI API key not found, skipping vector search')
      } else {
        const embeddingResponse = await fetch('https://api.openai.com/v1/embeddings', {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${openaiKey}`,
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            input: query,
            model: 'text-embedding-ada-002'
          })
        })

        if (embeddingResponse.ok) {
          const embeddingData = await embeddingResponse.json()
          queryEmbedding = embeddingData.data[0].embedding
        } else {
          console.warn('Failed to generate embedding:', await embeddingResponse.text())
        }
      }
    }

    // Vector Search
    if (type === 'vector' || type === 'hybrid') {
      if (queryEmbedding.length > 0) {
        let vectorQuery = supabase
          .from('products')
          .select(`
            *,
            category:taxonomy_nodes(id, name_th, name_en)
          `)
          .not('embedding', 'is', null)
          .limit(limit)
          .range(offset, offset + limit - 1)

        // Apply filters
        if (filters?.categories?.length) {
          vectorQuery = vectorQuery.in('category_id', filters.categories)
        }
        if (filters?.priceRange) {
          vectorQuery = vectorQuery
            .gte('price', filters.priceRange[0])
            .lte('price', filters.priceRange[1])
        }
        if (filters?.confidence) {
          vectorQuery = vectorQuery
            .gte('confidence_score', filters.confidence[0])
            .lte('confidence_score', filters.confidence[1])
        }

        const { data: vectorResults } = await vectorQuery

        if (vectorResults) {
          // Calculate cosine similarity
          const vectorMatches = vectorResults
            .map(product => {
              if (!product?.embedding || !Array.isArray(product.embedding)) return null
              
              const similarity = cosineSimilarity(queryEmbedding, product.embedding)
              if (similarity < 0.1) return null // Filter out very low similarities
              
              return {
                product,
                score: similarity,
                matchType: 'vector' as const,
                matchedTokens: [],
                explanation: `Semantic similarity: ${(similarity * 100).toFixed(1)}%`
              }
            })
            .filter(Boolean)
            .sort((a, b) => b!.score - a!.score)

          results.push(...vectorMatches)
        }
      }
    }

    // Text Search
    if (type === 'text' || type === 'hybrid') {
      const searchTerms = query.toLowerCase().split(/\s+/).filter(term => term.length > 1)
      
      let textQuery = supabase
        .from('products')
        .select(`
          *,
          category:taxonomy_nodes(id, name_th, name_en)
        `)
        .limit(limit)
        .range(offset, offset + limit - 1)

      // Build text search conditions
      const searchConditions = searchTerms.map(term => 
        `name_th.ilike.%${term}%,description.ilike.%${term}%,keywords.cs.{${term}}`
      ).join(',')

      if (searchConditions) {
        textQuery = textQuery.or(searchConditions)
      }

      // Apply filters
      if (filters?.categories?.length) {
        textQuery = textQuery.in('category_id', filters.categories)
      }
      if (filters?.priceRange) {
        textQuery = textQuery
          .gte('price', filters.priceRange[0])
          .lte('price', filters.priceRange[1])
      }
      if (filters?.confidence) {
        textQuery = textQuery
          .gte('confidence_score', filters.confidence[0])
          .lte('confidence_score', filters.confidence[1])
      }

      const { data: textResults } = await textQuery

      if (textResults) {
        const textMatches = textResults.map(product => {
          const matchedTokens: string[] = []
          let score = 0

          // Calculate text match score
          searchTerms.forEach(term => {
            if (product.name_th?.toLowerCase().includes(term)) {
              matchedTokens.push(term)
              score += 0.4
            }
            if (product.description?.toLowerCase().includes(term)) {
              matchedTokens.push(term)
              score += 0.3
            }
            if (product.keywords?.some((k: string) => k.toLowerCase().includes(term))) {
              matchedTokens.push(term)
              score += 0.3
            }
          })

          return {
            product,
            score: Math.min(score, 1.0),
            matchType: 'text' as const,
            matchedTokens: [...new Set(matchedTokens)],
            explanation: `Text matches: ${matchedTokens.length} terms`
          }
        }).filter(match => match.score > 0)

        results.push(...textMatches)
      }
    }

    // Merge and deduplicate results for hybrid search
    if (type === 'hybrid') {
      const productMap = new Map()
      
      results.forEach(result => {
        const productId = result.product.id
        if (productMap.has(productId)) {
          const existing = productMap.get(productId)
          // Combine scores with weighted average
          const combinedScore = (existing.score * 0.6) + (result.score * 0.4)
          const combinedTokens = [...new Set([...existing.matchedTokens, ...result.matchedTokens])]
          
          productMap.set(productId, {
            ...existing,
            score: combinedScore,
            matchType: 'hybrid' as const,
            matchedTokens: combinedTokens,
            explanation: `Hybrid match: semantic + text`
          })
        } else {
          productMap.set(productId, result)
        }
      })
      
      results = Array.from(productMap.values())
    }

    // Sort by score and apply limit
    results.sort((a, b) => b.score - a.score)
    results = results.slice(0, limit)

    const searchTime = Date.now() - startTime

    // Calculate stats
    const stats = {
      vectorMatches: results.filter(r => r.matchType === 'vector').length,
      textMatches: results.filter(r => r.matchType === 'text').length,
      hybridMatches: results.filter(r => r.matchType === 'hybrid').length,
    }

    return new Response(
      JSON.stringify({
        results,
        totalCount: results.length,
        searchTime,
        stats
      }),
      {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      }
    )

  } catch (error) {
    console.error('Search error:', error)
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

// Utility function to calculate cosine similarity
function cosineSimilarity(vecA: number[], vecB: number[]): number {
  if (vecA.length !== vecB.length) return 0

  let dotProduct = 0
  let normA = 0
  let normB = 0

  for (let i = 0; i < vecA.length; i++) {
    dotProduct += vecA[i] * vecB[i]
    normA += vecA[i] * vecA[i]
    normB += vecB[i] * vecB[i]
  }

  if (normA === 0 || normB === 0) return 0

  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB))
}
