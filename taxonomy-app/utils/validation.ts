/**
 * Request Validation Utility
 * ระบบ validate request body และ query parameters
 */

import { NextRequest } from 'next/server'
import { z, ZodSchema } from 'zod'
// import { logger } from './logger' // TODO: Fix import path
import { ApiErrors } from './error-handler'

/**
 * Validate request body กับ Zod schema
 */
export async function validateRequest<T>(
  request: NextRequest,
  schema: ZodSchema<T>
): Promise<T | { error: string; details: any }> {
  try {
    // Parse JSON body
    const body = await request.json()
    
    // Validate with schema
    const result = schema.safeParse(body)
    
    if (!result.success) {
      console.warn('Request validation failed', {
        errors: result.error.issues,
        body: body
      })
      
      return {
        error: 'Invalid request data',
        details: result.error.issues.map((err: any) => ({
          field: err.path.join('.'),
          message: err.message,
          code: err.code
        }))
      }
    }
    
    return result.data
  } catch (error: any) {
    console.error('Failed to parse request body', { error: error.message })
    
    if (error instanceof SyntaxError) {
      return {
        error: 'Invalid JSON format',
        details: { message: 'Request body must be valid JSON' }
      }
    }
    
    return {
      error: 'Failed to process request',
      details: { message: error.message }
    }
  }
}

/**
 * Validate query parameters
 */
export function validateQuery<T>(
  request: NextRequest,
  schema: ZodSchema<T>
): T | { error: string; details: any } {
  try {
    const { searchParams } = new URL(request.url)
    const queryParams = Object.fromEntries(searchParams.entries())
    
    const result = schema.safeParse(queryParams)
    
    if (!result.success) {
      console.warn('Query validation failed', {
        errors: result.error.issues,
        query: queryParams
      })
      
      return {
        error: 'Invalid query parameters',
        details: result.error.issues.map((err: any) => ({
          field: err.path.join('.'),
          message: err.message,
          code: err.code
        }))
      }
    }
    
    return result.data
  } catch (error: any) {
    console.error('Failed to validate query parameters', { error: error.message })
    
    return {
      error: 'Failed to process query parameters',
      details: { message: error.message }
    }
  }
}

/**
 * Common validation schemas
 */
export const CommonSchemas = {
  // UUID validation
  uuid: z.string().uuid('ต้องเป็น UUID ที่ถูกต้อง'),
  
  // Pagination
  pagination: z.object({
    limit: z.coerce.number().min(1).max(1000).default(100),
    offset: z.coerce.number().min(0).default(0),
    sort: z.string().optional(),
    order: z.enum(['asc', 'desc']).default('asc')
  }),
  
  // Search
  search: z.object({
    q: z.string().min(1).max(255).optional(),
    fields: z.string().optional(), // comma-separated fields
    exact: z.coerce.boolean().default(false)
  }),
  
  // Thai text
  thaiText: z.string()
    .min(1, 'ข้อความต้องไม่ว่าง')
    .max(255, 'ข้อความต้องไม่เกิน 255 ตัวอักษร')
    .regex(/[\u0e00-\u0e7f]/, 'ต้องมีอักษรไทยอย่างน้อย 1 ตัว'),
  
  // English text
  englishText: z.string()
    .min(1, 'Text cannot be empty')
    .max(255, 'Text must not exceed 255 characters')
    .regex(/^[a-zA-Z0-9\s\-_.,!?()]+$/, 'Only English characters, numbers and basic punctuation allowed'),
  
  // Keywords array
  keywords: z.array(z.string().min(1).max(50))
    .max(20, 'Keywords must not exceed 20 items'),
  
  // Metadata object
  metadata: z.record(z.string(), z.any())
    .refine(
      (obj) => JSON.stringify(obj).length <= 10000,
      'Metadata size must not exceed 10KB'
    )
}

/**
 * Sanitize input data
 */
export function sanitizeInput(input: any): any {
  if (typeof input === 'string') {
    return input
      .trim()
      .replace(/\s+/g, ' ') // Normalize whitespace
      .replace(/[<>]/g, '') // Remove potential HTML tags
  }
  
  if (Array.isArray(input)) {
    return input.map(sanitizeInput)
  }
  
  if (input && typeof input === 'object') {
    const sanitized: any = {}
    for (const [key, value] of Object.entries(input)) {
      sanitized[key] = sanitizeInput(value)
    }
    return sanitized
  }
  
  return input
}

/**
 * Validate file upload
 */
export const FileValidation = {
  image: z.object({
    name: z.string(),
    size: z.number().max(5 * 1024 * 1024, 'File size must not exceed 5MB'),
    type: z.string().regex(/^image\/(jpeg|jpg|png|gif|webp)$/, 'Only image files allowed')
  }),
  
  csv: z.object({
    name: z.string().endsWith('.csv', 'File must be CSV format'),
    size: z.number().max(50 * 1024 * 1024, 'File size must not exceed 50MB'),
    type: z.string().includes('csv')
  }),
  
  excel: z.object({
    name: z.string().regex(/\.(xlsx?|xls)$/, 'File must be Excel format'),
    size: z.number().max(50 * 1024 * 1024, 'File size must not exceed 50MB')
  })
}

/**
 * Custom validation helpers
 */
export const ValidationHelpers = {
  // Check if string contains Thai characters
  hasThai: (str: string): boolean => /[\u0e00-\u0e7f]/.test(str),
  
  // Check if string is valid email
  isEmail: (str: string): boolean => /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(str),
  
  // Check if string is valid URL
  isUrl: (str: string): boolean => {
    try {
      new URL(str)
      return true
    } catch {
      return false
    }
  },
  
  // Validate Thai phone number
  isThaiPhone: (str: string): boolean => /^(\+66|0)[0-9]{8,9}$/.test(str),
  
  // Check if array has unique values
  hasUniqueValues: <T>(arr: T[]): boolean => {
    return arr.length === new Set(arr).size
  }
}
