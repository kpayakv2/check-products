// Environment setup for integration tests
process.env.NODE_ENV = 'test'
process.env.NEXT_PUBLIC_APP_ENV = 'test'

// Load environment variables from .env.local (same as main app)
require('dotenv').config({ path: '.env.local' })

console.log('🔧 Integration test environment configured:', {
  NODE_ENV: process.env.NODE_ENV,
  SUPABASE_URL: process.env.NEXT_PUBLIC_SUPABASE_URL ? 'configured' : 'not configured',
  SUPABASE_KEY: process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY ? 'configured' : 'not configured'
})
