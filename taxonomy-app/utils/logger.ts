/**
 * Logging Utility
 * ระบบ logging ที่ครบถ้วนสำหรับ production
 */

export interface LogContext {
  [key: string]: any
}

export type LogLevel = 'debug' | 'info' | 'warn' | 'error'

export interface LogEntry {
  level: LogLevel
  message: string
  timestamp: string
  context?: LogContext
  error?: {
    message: string
    stack?: string
    name?: string
  }
}

class Logger {
  private isDevelopment = process.env.NODE_ENV === 'development'
  private logLevel: LogLevel = (process.env.LOG_LEVEL as LogLevel) || 'info'

  private shouldLog(level: LogLevel): boolean {
    const levels: Record<LogLevel, number> = {
      debug: 0,
      info: 1,
      warn: 2,
      error: 3
    }
    
    return levels[level] >= levels[this.logLevel]
  }

  private formatMessage(level: LogLevel, message: string, context?: LogContext): string {
    const timestamp = new Date().toISOString()
    const prefix = `[${timestamp}] ${level.toUpperCase()}`
    
    if (this.isDevelopment) {
      // Pretty format for development
      let formatted = `${prefix}: ${message}`
      if (context && Object.keys(context).length > 0) {
        formatted += `\n  Context: ${JSON.stringify(context, null, 2)}`
      }
      return formatted
    } else {
      // JSON format for production
      const logEntry: LogEntry = {
        level,
        message,
        timestamp,
        context
      }
      return JSON.stringify(logEntry)
    }
  }

  private log(level: LogLevel, message: string, context?: LogContext): void {
    if (!this.shouldLog(level)) return

    const formatted = this.formatMessage(level, message, context)
    
    switch (level) {
      case 'debug':
      case 'info':
        console.log(formatted)
        break
      case 'warn':
        console.warn(formatted)
        break
      case 'error':
        console.error(formatted)
        break
    }
  }

  debug(message: string, context?: LogContext): void {
    this.log('debug', message, context)
  }

  info(message: string, context?: LogContext): void {
    this.log('info', message, context)
  }

  warn(message: string, context?: LogContext): void {
    this.log('warn', message, context)
  }

  error(message: string, context?: LogContext): void
  error(message: string, error: Error, context?: LogContext): void
  error(message: string, errorOrContext?: Error | LogContext, context?: LogContext): void {
    let finalContext: LogContext = {}
    
    if (errorOrContext instanceof Error) {
      finalContext = {
        ...context,
        error: {
          message: errorOrContext.message,
          stack: errorOrContext.stack,
          name: errorOrContext.name
        }
      }
    } else if (errorOrContext) {
      finalContext = errorOrContext
    }
    
    this.log('error', message, finalContext)
  }

  // API request logging helpers
  apiRequest(method: string, path: string, context?: LogContext): void {
    this.info(`API ${method} ${path}`, {
      type: 'api_request',
      method,
      path,
      ...context
    })
  }

  apiResponse(method: string, path: string, status: number, duration?: number, context?: LogContext): void {
    const level = status >= 400 ? 'warn' : 'info'
    this.log(level, `API ${method} ${path} - ${status}`, {
      type: 'api_response',
      method,
      path,
      status,
      duration,
      ...context
    })
  }

  apiError(method: string, path: string, error: Error, context?: LogContext): void {
    this.error(`API ${method} ${path} failed`, error, {
      type: 'api_error',
      method,
      path,
      ...context
    })
  }

  // Database operation logging
  dbQuery(query: string, params?: any[], duration?: number): void {
    this.debug('Database query executed', {
      type: 'db_query',
      query: this.isDevelopment ? query : '[REDACTED]',
      paramCount: params?.length || 0,
      duration
    })
  }

  dbError(operation: string, error: Error, context?: LogContext): void {
    this.error(`Database ${operation} failed`, error, {
      type: 'db_error',
      operation,
      ...context
    })
  }

  // Performance logging
  performance(operation: string, duration: number, context?: LogContext): void {
    const level = duration > 1000 ? 'warn' : 'info'
    this.log(level, `Performance: ${operation} took ${duration}ms`, {
      type: 'performance',
      operation,
      duration,
      ...context
    })
  }

  // Security logging
  security(event: string, context?: LogContext): void {
    this.warn(`Security event: ${event}`, {
      type: 'security',
      event,
      ...context
    })
  }

  // Business logic logging
  business(event: string, context?: LogContext): void {
    this.info(`Business event: ${event}`, {
      type: 'business',
      event,
      ...context
    })
  }
}

// Create singleton instance
export const logger = new Logger()

// Performance measurement helper
export function measurePerformance<T>(
  operation: string,
  fn: () => Promise<T>,
  context?: LogContext
): Promise<T> {
  return new Promise(async (resolve, reject) => {
    const start = Date.now()
    
    try {
      const result = await fn()
      const duration = Date.now() - start
      logger.performance(operation, duration, context)
      resolve(result)
    } catch (error) {
      const duration = Date.now() - start
      logger.error(`${operation} failed after ${duration}ms`, error as Error, context)
      reject(error)
    }
  })
}

// Request ID generator for tracing
export function generateRequestId(): string {
  return `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
}

// Structured logging for different environments
export const StructuredLogger = {
  // Production-ready structured logging
  structured: (level: LogLevel, message: string, data: Record<string, any>) => {
    const entry = {
      '@timestamp': new Date().toISOString(),
      level,
      message,
      ...data,
      environment: process.env.NODE_ENV || 'development',
      service: 'taxonomy-app'
    }
    
    console.log(JSON.stringify(entry))
  },
  
  // Audit logging for compliance
  audit: (action: string, userId?: string, resource?: string, context?: LogContext) => {
    StructuredLogger.structured('info', `Audit: ${action}`, {
      type: 'audit',
      action,
      userId,
      resource,
      ...context
    })
  },
  
  // Metrics logging for monitoring
  metric: (name: string, value: number, unit?: string, tags?: Record<string, string>) => {
    StructuredLogger.structured('info', `Metric: ${name}`, {
      type: 'metric',
      name,
      value,
      unit,
      tags
    })
  }
}
