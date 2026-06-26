/// <reference types="https://deno.land/x/deno/lib/deno.ns.d.ts" />
import { serve } from "https://deno.land/std@0.168.0/http/server.ts"

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
}

interface DeduplicationRequest {
  oldProductsPath: string
  newProductsPath: string
  threshold?: number
  method?: 'hybrid' | 'tfidf' | 'semantic'
}

serve(async (req) => {
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders })
  }

  try {
    const { 
      oldProductsPath, 
      newProductsPath, 
      threshold = 0.75 
    }: DeduplicationRequest = await req.json()

    if (!oldProductsPath || !newProductsPath) {
      return new Response(
        JSON.stringify({ error: 'Missing file paths' }),
        { status: 400, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      )
    }

    // Call local FastAPI deduplication server
    // Using 127.0.0.1 instead of localhost for Windows compatibility as per constitution
    const API_URL = 'http://127.0.0.1:8000/api/v1/match/file-dedup'
    
    console.log(`[Proxy] Forwarding file deduplication request to FastAPI backend: ${API_URL}`)
    console.log(`- Old Path: ${oldProductsPath}`)
    console.log(`- New Path: ${newProductsPath}`)
    console.log(`- Threshold: ${threshold}`)

    const response = await fetch(API_URL, {
      method: 'POST',
      headers: { 
        'Content-Type': 'application/json' 
      },
      body: JSON.stringify({ 
        old_products_path: oldProductsPath,
        new_products_path: newProductsPath,
        threshold: threshold
      }),
    })

    if (!response.ok) {
      const errorText = await response.text()
      throw new Error(`FastAPI Deduplication API error: ${response.status} ${errorText}`)
    }

    const data = await response.json()
    console.log(`[Proxy] Successfully completed deduplication. Summary: ${data.summary || 'N/A'}`)

    return new Response(JSON.stringify(data), {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      status: 200,
    })

  } catch (error) {
    console.error('[Proxy] Deduplication error:', error)
    return new Response(JSON.stringify({ 
      error: error.message,
      help: 'Make sure FastAPI is running at http://127.0.0.1:8000 for local embeddings and ML models.'
    }), {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      status: 500,
    })
  }
})
