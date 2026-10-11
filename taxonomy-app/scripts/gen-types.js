#!/usr/bin/env node

/**
 * Two-Pillar Type Generation Script (ADR 0004)
 * Generates:
 * 1. types/database.types.ts from PostgreSQL (127.0.0.1:54325) via supabase CLI
 * 2. types/fastapi.types.ts from FastAPI (http://127.0.0.1:8000/openapi.json) via openapi-typescript
 */

const { execSync } = require('child_process')
const fs = require('fs')
const path = require('path')

const rootDir = path.resolve(__dirname, '..')
const typesDir = path.join(rootDir, 'types')

if (!fs.existsSync(typesDir)) {
  fs.mkdirSync(typesDir, { recursive: true })
}

console.log('🔄 Generating TypeScript definitions (ADR 0004: Two-Pillar SSOT)...\n')

// Pillar 1: Database Schema (PostgreSQL)
try {
  console.log('1. Generating types/database.types.ts from PostgreSQL (127.0.0.1:54325)...')
  const dbUrl = 'postgresql://postgres:postgres@127.0.0.1:54325/postgres?sslmode=disable'
  const dbTypes = execSync(`npx supabase gen types typescript --db-url "${dbUrl}"`, {
    cwd: rootDir,
    encoding: 'utf-8',
    maxBuffer: 10 * 1024 * 1024,
  })
  fs.writeFileSync(path.join(typesDir, 'database.types.ts'), dbTypes, 'utf-8')
  console.log('   ✅ types/database.types.ts generated successfully.')
} catch (err) {
  console.error('   ❌ Failed to generate database types:', err.message)
  process.exit(1)
}

// Pillar 2: AI Service Contracts (FastAPI / Pydantic)
try {
  console.log('2. Generating types/fastapi.types.ts from FastAPI (http://127.0.0.1:8000/openapi.json)...')
  const openapiUrl = 'http://127.0.0.1:8000/openapi.json'
  execSync(`npx openapi-typescript ${openapiUrl} -o types/fastapi.types.ts`, {
    cwd: rootDir,
    encoding: 'utf-8',
  })
  console.log('   ✅ types/fastapi.types.ts generated successfully.')
} catch (err) {
  console.error('   ❌ Failed to generate FastAPI types:', err.message)
  console.error('   ⚠️ Make sure FastAPI is running on 127.0.0.1:8000.')
  process.exit(1)
}

console.log('\n🎉 Type generation complete! Both pillars up to date.\n')
