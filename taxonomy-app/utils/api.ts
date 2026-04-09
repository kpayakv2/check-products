// API utilities for Supabase Edge Functions and external services
import { supabase } from './supabase'

export interface SearchRequest {
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

export interface SearchResponse {
  results: Array<{
    product: any
    score: number
    matchType: 'vector' | 'text' | 'hybrid'
    matchedTokens: string[]
    explanation: string
  }>
  totalCount: number
  searchTime: number
  stats: {
    vectorMatches: number
    textMatches: number
    hybridMatches: number
  }
}

export interface SuggestionRequest {
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

export interface SuggestionResponse {
  suggestions: Array<{
    categoryId: string
    categoryName: string
    confidence: number
    explanation: string
    matchedRules: string[]
    reasoning: string
  }>
  processingTime: number
  tokensUsed?: number
}

export interface EmbeddingRequest {
  texts: string[]
  model?: 'paraphrase-multilingual-MiniLM-L12-v2' | 'multilingual-e5-large'
}

export interface EmbeddingResponse {
  embeddings: number[][]
  model: string
  usage: {
    promptTokens: number
    totalTokens: number
  }
}

export class EdgeFunctionAPI {
  private static baseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL + '/functions/v1'
  
  // Hybrid Search Edge Function
  static async hybridSearch(request: SearchRequest): Promise<SearchResponse> {
    try {
      const { data, error } = await supabase.functions.invoke('hybrid-search', {
        body: request
      })

      if (error) throw error
      return data
    } catch (error) {
      console.error('Hybrid search error:', error)
      throw new Error('Failed to perform hybrid search')
    }
  }

  // Category Suggestion Edge Function
  static async getCategorySuggestions(request: SuggestionRequest): Promise<SuggestionResponse> {
    try {
      const { data, error } = await supabase.functions.invoke('category-suggestions', {
        body: request
      })

      if (error) throw error
      return data
    } catch (error) {
      console.error('Category suggestion error:', error)
      throw new Error('Failed to get category suggestions')
    }
  }

  // Text Embedding Edge Function
  static async generateEmbeddings(request: EmbeddingRequest): Promise<EmbeddingResponse> {
    try {
      const { data, error } = await supabase.functions.invoke('generate-embeddings', {
        body: request
      })

      if (error) throw error
      return data
    } catch (error) {
      console.error('Embedding generation error:', error)
      throw new Error('Failed to generate embeddings')
    }
  }

  // Similarity Matching Edge Function
  static async findSimilarProducts(productId: string, threshold: number = 0.8) {
    try {
      const { data, error } = await supabase.functions.invoke('similarity-matching', {
        body: { productId, threshold }
      })

      if (error) throw error
      return data
    } catch (error) {
      console.error('Similarity matching error:', error)
      throw new Error('Failed to find similar products')
    }
  }

  // Batch Processing Edge Function
  static async processBatch(batchId: string, operation: 'categorize' | 'deduplicate' | 'embed') {
    try {
      const { data, error } = await supabase.functions.invoke('batch-processing', {
        body: { batchId, operation }
      })

      if (error) throw error
      return data
    } catch (error) {
      console.error('Batch processing error:', error)
      throw new Error('Failed to process batch')
    }
  }

  // Thai Text Processing Edge Function
  static async processThaiText(text: string, operations: string[] = ['clean', 'tokenize', 'extract']) {
    try {
      const { data, error } = await supabase.functions.invoke('thai-text-processing', {
        body: { text, operations }
      })

      if (error) throw error
      return data
    } catch (error) {
      console.error('Thai text processing error:', error)
      throw new Error('Failed to process Thai text')
    }
  }
}

// Local API utilities for client-side operations
export class LocalAPI {
  // Client-side search with caching
  static async searchWithCache(query: string, filters: any = {}) {
    const cacheKey = `search_${JSON.stringify({ query, filters })}`
    const cached = localStorage.getItem(cacheKey)
    
    if (cached) {
      const { data, timestamp } = JSON.parse(cached)
      // Cache for 5 minutes
      if (Date.now() - timestamp < 5 * 60 * 1000) {
        return data
      }
    }

    try {
      const result = await EdgeFunctionAPI.hybridSearch({
        query,
        type: 'hybrid',
        filters,
        limit: 50
      })

      localStorage.setItem(cacheKey, JSON.stringify({
        data: result,
        timestamp: Date.now()
      }))

      return result
    } catch (error) {
      console.error('Search with cache error:', error)
      throw error
    }
  }

  // Debounced suggestions
  private static suggestionTimeout: NodeJS.Timeout | null = null
  
  static async getDebouncedSuggestions(
    text: string, 
    callback: (suggestions: SuggestionResponse) => void,
    delay: number = 300
  ) {
    if (this.suggestionTimeout) {
      clearTimeout(this.suggestionTimeout)
    }

    this.suggestionTimeout = setTimeout(async () => {
      try {
        const suggestions = await EdgeFunctionAPI.getCategorySuggestions({
          text,
          options: {
            maxSuggestions: 5,
            minConfidence: 0.3,
            includeExplanation: true
          }
        })
        callback(suggestions)
      } catch (error) {
        console.error('Debounced suggestions error:', error)
      }
    }, delay)
  }

  // Batch embedding generation with progress
  static async generateEmbeddingsBatch(
    texts: string[],
    onProgress?: (progress: number) => void,
    batchSize: number = 100
  ): Promise<number[][]> {
    const results: number[][] = []
    const totalBatches = Math.ceil(texts.length / batchSize)

    for (let i = 0; i < totalBatches; i++) {
      const batch = texts.slice(i * batchSize, (i + 1) * batchSize)
      
      try {
        const response = await EdgeFunctionAPI.generateEmbeddings({
          texts: batch
        })
        
        results.push(...response.embeddings)
        
        if (onProgress) {
          onProgress(((i + 1) / totalBatches) * 100)
        }
        
        // Small delay to avoid rate limiting
        if (i < totalBatches - 1) {
          await new Promise(resolve => setTimeout(resolve, 100))
        }
      } catch (error) {
        console.error(`Batch ${i + 1} failed:`, error)
        // Add empty embeddings for failed batch
        results.push(...Array(batch.length).fill(Array(768).fill(0)))
      }
    }

    return results
  }
}

// WebSocket-like real-time updates using Supabase subscriptions
export class RealtimeAPI {
  private static subscriptions: Map<string, any> = new Map()

  static subscribeToTable(
    table: string,
    callback: (payload: any) => void,
    filter?: string
  ) {
    const subscription = supabase
      .channel(`realtime-${table}`)
      .on('postgres_changes', 
        { 
          event: '*', 
          schema: 'public', 
          table,
          filter 
        }, 
        callback
      )
      .subscribe()

    this.subscriptions.set(table, subscription)
    return subscription
  }

  static unsubscribeFromTable(table: string) {
    const subscription = this.subscriptions.get(table)
    if (subscription) {
      subscription.unsubscribe()
      this.subscriptions.delete(table)
    }
  }

  static unsubscribeAll() {
    this.subscriptions.forEach(subscription => {
      subscription.unsubscribe()
    })
    this.subscriptions.clear()
  }
}

// Error handling and retry logic
export class APIErrorHandler {
  static async withRetry<T>(
    operation: () => Promise<T>,
    maxRetries: number = 3,
    delay: number = 1000
  ): Promise<T> {
    let lastError: Error

    for (let attempt = 1; attempt <= maxRetries; attempt++) {
      try {
        return await operation()
      } catch (error) {
        lastError = error as Error
        
        if (attempt === maxRetries) {
          throw lastError
        }

        // Exponential backoff
        await new Promise(resolve => 
          setTimeout(resolve, delay * Math.pow(2, attempt - 1))
        )
      }
    }

    throw lastError!
  }

  static handleAPIError(error: any): string {
    if (error.message?.includes('rate limit')) {
      return 'API rate limit exceeded. Please try again later.'
    }
    
    if (error.message?.includes('network')) {
      return 'Network error. Please check your connection.'
    }
    
    if (error.message?.includes('unauthorized')) {
      return 'Authentication error. Please check your API keys.'
    }
    
    return error.message || 'An unexpected error occurred.'
  }
}

// Performance monitoring
export class APIMetrics {
  private static metrics: Map<string, number[]> = new Map()

  static recordAPICall(endpoint: string, duration: number) {
    if (!this.metrics.has(endpoint)) {
      this.metrics.set(endpoint, [])
    }
    
    const times = this.metrics.get(endpoint)!
    times.push(duration)
    
    // Keep only last 100 calls
    if (times.length > 100) {
      times.shift()
    }
  }

  static getAverageResponseTime(endpoint: string): number {
    const times = this.metrics.get(endpoint)
    if (!times || times.length === 0) return 0
    
    return times.reduce((sum, time) => sum + time, 0) / times.length
  }

  static getAllMetrics() {
    const result: Record<string, { average: number, calls: number }> = {}
    
    this.metrics.forEach((times, endpoint) => {
      result[endpoint] = {
        average: this.getAverageResponseTime(endpoint),
        calls: times.length
      }
    })
    
    return result
  }
}
