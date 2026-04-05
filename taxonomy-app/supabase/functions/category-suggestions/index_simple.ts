// Simple category suggestions using only taxonomy_nodes keywords
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2.39.0'

const supabaseUrl = Deno.env.get('SUPABASE_URL')!
const supabaseServiceRoleKey = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY')!

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
}

Deno.serve(async (req) => {
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders })
  }

  try {
    const { text, options = {} } = await req.json()
    const { maxSuggestions = 5, minConfidence = 0.3, includeExplanation = true } = options

    if (!text?.trim()) {
      return new Response(
        JSON.stringify({ error: 'Text is required' }),
        { status: 400, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      )
    }

    const supabase = createClient(supabaseUrl, supabaseServiceRoleKey)
    
    // Get categories with keywords
    const { data: categories, error } = await supabase
      .from('taxonomy_nodes')
      .select('id, code, name_th, name_en, keywords, description, level')
      .eq('is_active', true)
      .not('keywords', 'is', null)

    if (error) {
      console.error('Database error:', error)
      return new Response(
        JSON.stringify({ error: 'Database error', details: error.message }),
        { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      )
    }

    if (!categories || categories.length === 0) {
      return new Response(
        JSON.stringify({ 
          suggestions: [],
          processingTime: 0,
          tokensUsed: 0,
          debug: 'No categories found with keywords'
        }),
        { headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      )
    }

    const startTime = Date.now()
    const cleanText = text.toLowerCase().trim()
    const tokens = cleanText.split(/[\s\-\(\)\[\]\/\\,\.]+/).filter(t => t.length >= 2)
    
    console.log(`Processing text: "${text}", clean: "${cleanText}"`)
    console.log(`Found ${categories.length} categories with keywords`)

    const suggestions: any[] = []

    categories.forEach((category: any) => {
      if (!category.keywords || !Array.isArray(category.keywords)) return

      let score = 0
      const matchedKeywords: string[] = []

      category.keywords.forEach((keyword: string) => {
        const keywordLower = keyword.toLowerCase()
        
        // Exact match
        if (cleanText.includes(keywordLower)) {
          score += 1.0
          matchedKeywords.push(keyword)
        }
        // Token match
        else if (tokens.some(token => token === keywordLower)) {
          score += 0.8
          matchedKeywords.push(keyword)
        }
        // Partial match
        else if (keywordLower.length >= 3 && tokens.some(token => token.includes(keywordLower) || keywordLower.includes(token))) {
          score += 0.3
          matchedKeywords.push(keyword)
        }
      })

      if (score > 0) {
        const confidence = Math.min(score / category.keywords.length, 1.0)
        if (confidence >= minConfidence) {
          suggestions.push({
            categoryId: category.id,
            categoryCode: category.code,
            categoryName: category.name_th || category.name_en,
            confidence: Math.round(confidence * 100) / 100,
            matchedKeywords: matchedKeywords,
            ...(includeExplanation && {
              explanation: `Matched ${matchedKeywords.length} keywords: ${matchedKeywords.join(', ')}`
            })
          })
        }
      }
    })

    // Sort by confidence and limit results
    suggestions.sort((a, b) => b.confidence - a.confidence)
    const finalSuggestions = suggestions.slice(0, maxSuggestions)

    const processingTime = Date.now() - startTime

    return new Response(
      JSON.stringify({
        suggestions: finalSuggestions,
        processingTime,
        tokensUsed: tokens.length,
        debug: {
          cleanText,
          tokens,
          categoriesFound: categories.length,
          totalMatches: suggestions.length
        }
      }),
      { headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    )

  } catch (error) {
    console.error('Error:', error)
    return new Response(
      JSON.stringify({ error: error.message }),
      { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    )
  }
})