/// <reference types="https://deno.land/x/deno/lib/deno.ns.d.ts" />
import { serve } from "https://deno.land/std@0.168.0/http/server.ts"

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
}

interface EmbeddingRequest {
  texts: string[]
  model?: 'sentence-transformer' | 'transformers-js'
}

/**
 * Generate Embeddings using Local Models
 * 
 * Options:
 * 1. sentence-transformer (via FastAPI) - Same as FastAPI backend
 * 2. transformers-js (Deno native) - Pure JavaScript implementation
 */
serve(async (req) => {
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders })
  }

  try {
    const { texts, model = 'sentence-transformer' }: EmbeddingRequest = await req.json()

    if (!texts || !Array.isArray(texts) || texts.length === 0) {
      return new Response(
        JSON.stringify({ error: 'Texts array is required' }),
        { status: 400, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      )
    }

    if (texts.length > 100) {
      return new Response(
        JSON.stringify({ error: 'Maximum 100 texts per request' }),
        { status: 400, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      )
    }

    const startTime = Date.now()
    let embeddings: number[][] = []
    let dimension = 0

    if (model === 'sentence-transformer') {
      // Option 1: Use FastAPI backend (Same model as FastAPI)
      // Model: paraphrase-multilingual-MiniLM-L12-v2 (384-dim)
      
      const fastapiUrl = Deno.env.get('FASTAPI_URL') || 'http://host.docker.internal:8000'
      console.log('FastAPI URL:', fastapiUrl)
      
      // Call FastAPI single endpoint for each text
      embeddings = []
      for (const text of texts) {
        const response = await fetch(`${fastapiUrl}/api/embed`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            text: text
          })
        })

        if (!response.ok) {
          const error = await response.text()
          throw new Error(`FastAPI error: ${error}`)
        }

        const data = await response.json()
        embeddings.push(data.embedding)
        dimension = 384 // Local model dimension
      }

    } else if (model === 'transformers-js') {
      // Option 2: Use Transformers.js (Deno native)
      // Note: Requires @xenova/transformers package
      // This is a JavaScript implementation of transformers
      
      throw new Error('transformers-js not implemented yet. Use sentence-transformer model.')
      
      // Future implementation:
      // import { pipeline } from 'npm:@xenova/transformers@2.6.0'
      // const extractor = await pipeline('feature-extraction', 'Xenova/paraphrase-multilingual-MiniLM-L12-v2')
      // const output = await extractor(texts, { pooling: 'mean', normalize: true })
      // embeddings = output.tolist()
    }

    const processingTime = Date.now() - startTime

    return new Response(
      JSON.stringify({
        embeddings,
        model: 'paraphrase-multilingual-MiniLM-L12-v2',
        dimension: dimension || 384,
        backend: model === 'sentence-transformer' ? 'FastAPI' : 'Deno',
        usage: {
          textCount: texts.length,
          totalTokens: texts.reduce((sum, text) => sum + text.split(' ').length, 0)
        },
        processingTime,
        cost: 0, // FREE! Local model
        note: 'Using local model - same as FastAPI backend'
      }),
      {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      }
    )

  } catch (error) {
    console.error('Local embedding generation error:', error)
    const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred'
    return new Response(
      JSON.stringify({ 
        error: errorMessage,
        help: 'Make sure FastAPI is running at http://localhost:8000 or set FASTAPI_URL env variable'
      }),
      {
        status: 500,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      }
    )
  }
})
