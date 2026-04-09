import { serve } from "https://deno.land/std@0.168.0/http/server.ts"

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
}

serve(async (req) => {
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders })
  }

  try {
    const { text } = await req.json()
    
    if (!text) {
      return new Response(JSON.stringify({ error: 'Text is required' }), {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        status: 400,
      })
    }

    // Call local FastAPI embedding server
    // Using 127.0.0.1 instead of localhost for Windows compatibility
    const API_URL = 'http://127.0.0.1:8000/api/embed'
    
    const response = await fetch(API_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ texts: [text] }),
    })

    if (!response.ok) {
      const errorText = await response.text()
      throw new Error(`Embedding API error: ${response.status} ${errorText}`)
    }

    const data = await response.json()
    const embedding = data.embeddings[0]

    return new Response(JSON.stringify({ embedding }), {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      status: 200,
    })

  } catch (error) {
    return new Response(JSON.stringify({ 
      error: error.message,
      help: 'Make sure FastAPI is running at http://127.0.0.1:8000 for local embeddings'
    }), {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      status: 500,
    })
  }
})
