/**
 * Error Handling Utility
 * ระบบจัดการ error แบบครบถ้วนสำหรับ API routes
 */

import { NextResponse } from 'next/server'
// import { logger } from './logger' // TODO: Fix import path

export interface ApiError extends Error {
  status?: number
  code?: string
  details?: any
}

export interface ErrorResponse {
  success: false
  error: string
  message?: string
  code?: string
  details?: any
  timestamp: string
  requestId?: string
}

/**
 * สร้าง API Error พร้อม status code
 */
export function createApiError(
  message: string,
  status: number = 500,
  code?: string,
  details?: any
): ApiError {
  const error = new Error(message) as ApiError
  error.status = status
  error.code = code
  error.details = details
  return error
}

/**
 * แปลง Error เป็น ErrorResponse
 */
export function formatErrorResponse(
  error: any,
  requestId?: string
): ErrorResponse {
  const timestamp = new Date().toISOString()
  
  // Handle known API errors
  if (error.status) {
    return {
      success: false,
      error: error.message || 'An error occurred',
      code: error.code,
      details: error.details,
      timestamp,
      requestId
    }
  }
  
  // Handle Zod validation errors
  if (error.name === 'ZodError') {
    return {
      success: false,
      error: 'Validation failed',
      code: 'VALIDATION_ERROR',
      details: error.errors,
      timestamp,
      requestId
    }
  }
  
  // Handle database errors
  if (error.code && error.code.startsWith('23')) { // PostgreSQL constraint errors
    return {
      success: false,
      error: 'Database constraint violation',
      code: 'DATABASE_ERROR',
      details: process.env.NODE_ENV === 'development' ? error.detail : undefined,
      timestamp,
      requestId
    }
  }
  
  // Handle network/connection errors
  if (error.code === 'ECONNREFUSED' || error.code === 'ENOTFOUND') {
    return {
      success: false,
      error: 'Service unavailable',
      code: 'SERVICE_UNAVAILABLE',
      timestamp,
      requestId
    }
  }
  
  // Handle generic errors
  return {
    success: false,
    error: process.env.NODE_ENV === 'development' 
      ? error.message || 'Internal server error'
      : 'Internal server error',
    code: 'INTERNAL_ERROR',
    timestamp,
    requestId
  }
}

/**
 * Error handling wrapper สำหรับ API routes
 */
export function withErrorHandling<T>(
  handler: () => Promise<T>,
  context?: string
): Promise<T | NextResponse> {
  return (async (): Promise<T | NextResponse> => {
    const requestId = generateRequestId()
    
    try {
      console.info(`API request started`, { context, requestId })
      const result = await handler()
      console.info(`API request completed successfully`, { context, requestId })
      return result
    } catch (error: any) {
      console.error(`API request failed`, {
        context,
        requestId,
        error: error.message,
        stack: error.stack,
        status: error.status
      })
      
      const errorResponse = formatErrorResponse(error, requestId)
      const status = error.status || 500
      
      return NextResponse.json(errorResponse, { status })
    }
  })()
}

/**
 * Async error handler สำหรับ middleware
 */
export async function handleAsyncError<T>(
  promise: Promise<T>,
  context?: string
): Promise<[T | null, ApiError | null]> {
  try {
    const result = await promise
    return [result, null]
  } catch (error: any) {
    console.error(`Async operation failed`, {
      context,
      error: error.message,
      stack: error.stack
    })
    
    return [null, error]
  }
}

/**
 * สร้าง Request ID สำหรับ tracking
 */
function generateRequestId(): string {
  return `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
}

/**
 * Common API Errors
 */
export const ApiErrors = {
  BadRequest: (message: string, details?: any) => 
    createApiError(message, 400, 'BAD_REQUEST', details),
  
  Unauthorized: (message: string = 'Unauthorized') => 
    createApiError(message, 401, 'UNAUTHORIZED'),
  
  Forbidden: (message: string = 'Forbidden') => 
    createApiError(message, 403, 'FORBIDDEN'),
  
  NotFound: (message: string = 'Not found') => 
    createApiError(message, 404, 'NOT_FOUND'),
  
  Conflict: (message: string, details?: any) => 
    createApiError(message, 409, 'CONFLICT', details),
  
  ValidationError: (message: string, details?: any) => 
    createApiError(message, 422, 'VALIDATION_ERROR', details),
  
  TooManyRequests: (message: string = 'Too many requests') => 
    createApiError(message, 429, 'TOO_MANY_REQUESTS'),
  
  InternalError: (message: string = 'Internal server error', details?: any) => 
    createApiError(message, 500, 'INTERNAL_ERROR', details),
  
  ServiceUnavailable: (message: string = 'Service unavailable') => 
    createApiError(message, 503, 'SERVICE_UNAVAILABLE')
}

/**
 * Error boundary สำหรับ React components (ถ้าต้องการใช้)
 */
export class ApiErrorBoundary extends Error {
  constructor(
    public error: ApiError,
    public context?: string
  ) {
    super(error.message)
    this.name = 'ApiErrorBoundary'
  }
}
