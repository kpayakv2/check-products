/// <reference types="https://deno.land/x/deno/lib/deno.ns.d.ts" />
import { serve } from "https://deno.land/std@0.168.0/http/server.ts"
import { createClient } from '@supabase/supabase-js'

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
}

interface SuggestionRequest {
  text: string
  context?: {
    category?: string
    brand?: string
    attributes?: Record<string, any>
  }
  options?: {
    maxSuggestions?: number
    minConfidence?: number
    includeExplanation?: boolean
  }
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

    const { 
      text, 
      context = {}, 
      options = {} 
    }: SuggestionRequest = await req.json()

    const {
      maxSuggestions = 5,
      minConfidence = 0.3,
      includeExplanation = true
    } = options

    if (!text?.trim()) {
      return new Response(
        JSON.stringify({ error: 'Text is required' }),
        { status: 400, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      )
    }

    const startTime = Date.now()
    const suggestions: any[] = []

    // Clean and tokenize text
    const cleanText = text.toLowerCase()
      .replace(/[^\u0E00-\u0E7Fa-zA-Z0-9\s\-\.\(\)]/g, '')
      .replace(/\s+/g, ' ')
      .trim()

    const tokens = cleanText.split(/[\s\-\(\)\[\]\/\\,\.]+/)
      .filter(token => token.length >= 2)

    // Get all taxonomy nodes with keywords
    const { data: categories, error: categoriesError } = await supabase
      .from('taxonomy_nodes')
      .select('id, code, name_th, name_en, keywords, description, level')
      .eq('is_active', true)
      .not('keywords', 'is', null)

    if (categoriesError) {
      console.error('Error fetching categories:', categoriesError)
      return new Response(
        JSON.stringify({ 
          error: 'Failed to fetch categories',
          details: categoriesError.message 
        }),
        { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      )
    }

    console.log(`Found ${categories?.length || 0} categories with keywords`)

    // Skip keyword_rules and regex_rules tables for now - focus on taxonomy_nodes keywords only

    // Score categories based on keyword matching
    const categoryScores = new Map<string, {
      category: any,
      score: number,
      matchedKeywords: string[],
      matchedRules: string[],
      reasoning: string[]
    }>()

    // Process taxonomy node keywords
    if (categories && Array.isArray(categories)) {
      categories.forEach(category => {
        if (!category?.keywords || !Array.isArray(category.keywords) || category.keywords.length === 0) return

        let score = 0
        const matchedKeywords: string[] = []
        const reasoning: string[] = []

        // Check token matches
        tokens.forEach(token => {
          category.keywords.forEach((keyword: string) => {
            const keywordLower = keyword.toLowerCase()
            if (token.includes(keywordLower) || keywordLower.includes(token)) {
              score += 0.3
              matchedKeywords.push(keyword)
              reasoning.push(`Token "${token}" matches keyword "${keyword}"`)
            }
          })
        })

        // Bonus for exact matches
        category.keywords.forEach((keyword: string) => {
          if (cleanText.includes(keyword.toLowerCase())) {
            score += 0.5
            if (!matchedKeywords.includes(keyword)) {
              matchedKeywords.push(keyword)
            }
            reasoning.push(`Exact match for keyword "${keyword}"`)
          }
        })

        // Context bonuses
        if (context.brand && category.keywords.some((k: string) => 
          k.toLowerCase().includes(context.brand!.toLowerCase())
        )) {
          score += 0.2
          reasoning.push(`Brand context match`)
        }

        if (score > 0) {
          categoryScores.set(category.id, {
            category,
            score: Math.min(score, 1.0),
            matchedKeywords,
            matchedRules: [],
            reasoning
          })
        }
      })
    }

    // Process keyword rules
    if (keywordRules && Array.isArray(keywordRules)) {
      keywordRules.forEach(rule => {
        if (!rule?.category || !rule?.keywords || !Array.isArray(rule.keywords)) return

        let ruleScore = 0
        const matchedKeywords: string[] = []
        const reasoning: string[] = []

        rule.keywords.forEach((keyword: string) => {
          const keywordLower = keyword.toLowerCase()
          let isMatch = false

          switch (rule.match_type) {
            case 'exact':
              isMatch = cleanText === keywordLower
              break
            case 'contains':
              isMatch = cleanText.includes(keywordLower)
              break
            case 'fuzzy':
              // Simple fuzzy matching
              isMatch = tokens.some(token => {
                const distance = levenshteinDistance(token, keywordLower)
                return distance <= Math.max(1, Math.floor(keywordLower.length * 0.2))
              })
              break
          }

          if (isMatch) {
            ruleScore += rule.confidence_score / rule.keywords.length
            matchedKeywords.push(keyword)
            reasoning.push(`Rule "${rule.name}" matched keyword "${keyword}" (${rule.match_type})`)
          }
        })

        if (ruleScore > 0) {
          const categoryId = rule.category.id
          const existing = categoryScores.get(categoryId)
          
          if (existing) {
            existing.score = Math.min(existing.score + ruleScore * 0.8, 1.0)
            existing.matchedKeywords.push(...matchedKeywords)
            existing.matchedRules.push(rule.name)
            existing.reasoning.push(...reasoning)
          } else {
            categoryScores.set(categoryId, {
              category: rule.category,
              score: Math.min(ruleScore, 1.0),
              matchedKeywords,
              matchedRules: [rule.name],
              reasoning
            })
          }
        }
      })
    }

    // Process regex rules
    if (regexRules && Array.isArray(regexRules)) {
      regexRules.forEach(rule => {
        if (!rule?.category || !rule?.pattern) return

        try {
          const regex = new RegExp(rule.pattern, rule.flags || 'gi')
          const matches = cleanText.match(regex)

          if (matches && matches.length > 0) {
            const ruleScore = rule.confidence_score || 0.7
            const categoryId = rule.category.id
            const existing = categoryScores.get(categoryId)
            const reasoning = [`Regex rule "${rule.name}" matched: ${matches.join(', ')}`]

            if (existing) {
              existing.score = Math.min(existing.score + ruleScore * 0.6, 1.0)
              existing.matchedRules.push(rule.name)
              existing.reasoning.push(...reasoning)
            } else {
              categoryScores.set(categoryId, {
                category: rule.category,
                score: Math.min(ruleScore, 1.0),
                matchedKeywords: [],
                matchedRules: [rule.name],
                reasoning
              })
            }
          }
        } catch (error) {
          console.error(`Regex rule "${rule.name}" error:`, error)
        }
      })
    }

    // Convert to suggestions array and sort by score
    const sortedSuggestions = Array.from(categoryScores.values())
      .filter(item => item.score >= minConfidence)
      .sort((a, b) => b.score - a.score)
      .slice(0, maxSuggestions)
      .map(item => ({
        categoryId: item.category.id,
        categoryName: item.category.name_th,
        confidence: item.score,
        explanation: includeExplanation ? 
          `Matched ${item.matchedKeywords.length} keywords${item.matchedRules.length > 0 ? ` and ${item.matchedRules.length} rules` : ''}` :
          '',
        matchedRules: item.matchedRules,
        reasoning: includeExplanation ? item.reasoning.join('; ') : ''
      }))

    const processingTime = Date.now() - startTime

    return new Response(
      JSON.stringify({
        suggestions: sortedSuggestions,
        processingTime,
        tokensUsed: tokens.length
      }),
      {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      }
    )

  } catch (error) {
    console.error('Category suggestion error:', error)
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

// Utility function for fuzzy matching
function levenshteinDistance(str1: string, str2: string): number {
  const matrix = Array(str2.length + 1).fill(null).map(() => Array(str1.length + 1).fill(null))

  for (let i = 0; i <= str1.length; i++) matrix[0][i] = i
  for (let j = 0; j <= str2.length; j++) matrix[j][0] = j

  for (let j = 1; j <= str2.length; j++) {
    for (let i = 1; i <= str1.length; i++) {
      const indicator = str1[i - 1] === str2[j - 1] ? 0 : 1
      matrix[j][i] = Math.min(
        matrix[j][i - 1] + 1,
        matrix[j - 1][i] + 1,
        matrix[j - 1][i - 1] + indicator
      )
    }
  }

  return matrix[str2.length][str1.length]
}
