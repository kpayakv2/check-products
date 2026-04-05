/**
 * Rate Limiting Utility
 * ใช้สำหรับจำกัดจำนวนคำขอ API เพื่อป้องกัน abuse
 */

interface RateLimitOptions {
  interval: number // ช่วงเวลาในหน่วย milliseconds
  uniqueTokenPerInterval: number // จำนวน unique tokens สูงสุดต่อช่วงเวลา
}

interface RateLimitResult {
  limit: number
  remaining: number
  reset: Date
}

class RateLimiter {
  private tokens: Map<string, { count: number; resetTime: number }> = new Map()
  private options: RateLimitOptions

  constructor(options: RateLimitOptions) {
    this.options = options
  }

  async check(limit: number, token: string): Promise<RateLimitResult> {
    const now = Date.now()
    const key = `${token}-${Math.floor(now / this.options.interval)}`
    
    // Clean up old entries
    this.cleanup(now)
    
    const current = this.tokens.get(key) || { count: 0, resetTime: now + this.options.interval }
    
    if (current.count >= limit) {
      const error = new Error('Rate limit exceeded')
      ;(error as any).status = 429
      throw error
    }
    
    // Update count
    current.count++
    this.tokens.set(key, current)
    
    return {
      limit,
      remaining: Math.max(0, limit - current.count),
      reset: new Date(current.resetTime)
    }
  }

  private cleanup(now: number) {
    // Remove expired entries
    for (const [key, value] of this.tokens.entries()) {
      if (value.resetTime < now) {
        this.tokens.delete(key)
      }
    }
    
    // If too many entries, remove oldest
    if (this.tokens.size > this.options.uniqueTokenPerInterval) {
      const entries = Array.from(this.tokens.entries())
      entries.sort((a, b) => a[1].resetTime - b[1].resetTime)
      
      const toRemove = entries.slice(0, entries.length - this.options.uniqueTokenPerInterval)
      for (const [key] of toRemove) {
        this.tokens.delete(key)
      }
    }
  }
}

// Factory function
export function rateLimit(options: RateLimitOptions) {
  return new RateLimiter(options)
}

// Default rate limiter instances
export const defaultLimiter = rateLimit({
  interval: 60 * 1000, // 1 minute
  uniqueTokenPerInterval: 500
})

export const strictLimiter = rateLimit({
  interval: 60 * 1000, // 1 minute  
  uniqueTokenPerInterval: 100
})

// Helper to get client IP
export function getClientIP(request: Request): string {
  const forwarded = request.headers.get('x-forwarded-for')
  const realIp = request.headers.get('x-real-ip')
  
  if (forwarded) {
    return forwarded.split(',')[0].trim()
  }
  
  if (realIp) {
    return realIp.trim()
  }
  
  return 'unknown'
}

// Rate limit middleware helper
export function withRateLimit(
  limiter: RateLimiter,
  limit: number,
  keyGenerator?: (request: Request) => string
) {
  return async (request: Request) => {
    const key = keyGenerator ? keyGenerator(request) : getClientIP(request)
    
    try {
      const result = await limiter.check(limit, key)
      return { success: true, result }
    } catch (error: any) {
      return { 
        success: false, 
        error: error.message,
        status: error.status || 429
      }
    }
  }
}
