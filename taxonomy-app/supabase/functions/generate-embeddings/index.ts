/// <reference types="https://deno.land/x/deno/lib/deno.ns.d.ts" />
import { serve } from "https://deno.land/std@0.168.0/http/server.ts"

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
}

interface EmbeddingRequest {
  texts: string[]
  model?: 'text-embedding-ada-002' | 'multilingual-e5-large'
}

serve(async (req) => {
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders })
  }

  try {
    const { texts, model = 'text-embedding-ada-002' }: EmbeddingRequest = await req.json()

    if (!texts || !Array.isArray(texts) || texts.length === 0) {
      return new Response(
        JSON.stringify({ error: 'Texts array is required' }),
        { status: 400, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      )
    }

    // Limit batch size to prevent timeout
    if (texts.length > 100) {
      return new Response(
        JSON.stringify({ error: 'Maximum 100 texts per request' }),
        { status: 400, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      )
    }

    const startTime = Date.now()
    let embeddings: number[][] = []
    let totalTokens = 0

    if (model === 'text-embedding-ada-002') {
      // Check OpenAI API key
      const openaiKey = Deno.env.get('OPENAI_API_KEY')
      if (!openaiKey) {
        throw new Error('OpenAI API key not configured')
      }

      // OpenAI API
      const response = await fetch('https://api.openai.com/v1/embeddings', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${openaiKey}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          input: texts,
          model: 'text-embedding-ada-002'
        })
      })

      if (!response.ok) {
        const error = await response.text()
        throw new Error(`OpenAI API error: ${error}`)
      }

      const data = await response.json()
      embeddings = data.data.map((item: any) => item.embedding)
      totalTokens = data.usage.total_tokens

    } else if (model === 'multilingual-e5-large') {
      // Check Hugging Face API key
      const hfKey = Deno.env.get('HUGGINGFACE_API_KEY')
      if (!hfKey) {
        throw new Error('Hugging Face API key not configured')
      }

      // Hugging Face API
      const response = await fetch(
        'https://api-inference.huggingface.co/models/intfloat/multilingual-e5-large',
        {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${hfKey}`,
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            inputs: texts,
            options: { wait_for_model: true }
          })
        }
      )

      if (!response.ok) {
        const error = await response.text()
        throw new Error(`Hugging Face API error: ${error}`)
      }

      const data = await response.json()
      embeddings = Array.isArray(data[0]) ? data : [data]
      totalTokens = texts.reduce((sum, text) => sum + text.split(' ').length, 0)
    }

    const processingTime = Date.now() - startTime

    return new Response(
      JSON.stringify({
        embeddings,
        model,
        usage: {
          promptTokens: totalTokens,
          totalTokens
        },
        processingTime
      }),
      {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      }
    )

  } catch (error) {
    console.error('Embedding generation error:', error)
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
