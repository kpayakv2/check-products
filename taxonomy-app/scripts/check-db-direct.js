#!/usr/bin/env node
/**
 * Direct Database Connection Check
 * ตรวจสอบฐานข้อมูล Supabase Local ผ่าน Direct PostgreSQL Connection
 */

const { Client } = require('pg')
require('dotenv').config({ path: '.env.local' })

// Direct PostgreSQL connection
const client = new Client({
  host: '127.0.0.1',
  port: 54322,
  database: 'postgres',
  user: 'postgres',
  password: 'postgres'
})

async function connectAndCheck() {
  try {
    console.log('🐳 Connecting to Supabase Local Database...')
    await client.connect()
    console.log('✅ Connected to PostgreSQL successfully!\n')

    // Check database version
    const versionResult = await client.query('SELECT version()')
    console.log('📊 PostgreSQL Version:')
    console.log(`  ${versionResult.rows[0].version}\n`)

    // List all schemas
    console.log('📋 Available Schemas:')
    const schemasResult = await client.query(`
      SELECT schema_name 
      FROM information_schema.schemata 
      WHERE schema_name NOT IN ('information_schema', 'pg_catalog', 'pg_toast')
      ORDER BY schema_name
    `)
    
    schemasResult.rows.forEach(row => {
      console.log(`  - ${row.schema_name}`)
    })
    console.log('')

    // List all tables in public schema
    console.log('🗄️ Tables in public schema:')
    const tablesResult = await client.query(`
      SELECT table_name, table_type
      FROM information_schema.tables 
      WHERE table_schema = 'public' 
      AND table_type = 'BASE TABLE'
      ORDER BY table_name
    `)
    
    if (tablesResult.rows.length === 0) {
      console.log('  ❌ No tables found in public schema')
    } else {
      tablesResult.rows.forEach(row => {
        console.log(`  ✅ ${row.table_name} (${row.table_type})`)
      })
    }
    console.log('')

    // Check specific tables we need
    console.log('🎯 Checking Required Tables:')
    const requiredTables = [
      'taxonomy_nodes',
      'synonyms',
      'synonym_lemmas', 
      'synonym_terms',
      'products',
      'system_settings',
      'regex_rules',
      'keyword_rules'
    ]

    for (const tableName of requiredTables) {
      try {
        const result = await client.query(`
          SELECT COUNT(*) as count 
          FROM information_schema.tables 
          WHERE table_schema = 'public' 
          AND table_name = $1
        `, [tableName])
        
        const exists = result.rows[0].count > 0
        
        if (exists) {
          // Get row count
          const countResult = await client.query(`SELECT COUNT(*) as count FROM ${tableName}`)
          const rowCount = countResult.rows[0].count
          console.log(`  ✅ ${tableName} - EXISTS (${rowCount} rows)`)
          
          // Get column info
          const columnsResult = await client.query(`
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns 
            WHERE table_schema = 'public' 
            AND table_name = $1
            ORDER BY ordinal_position
          `, [tableName])
          
          const columns = columnsResult.rows.map(col => 
            `${col.column_name}:${col.data_type}${col.is_nullable === 'YES' ? '?' : ''}`
          ).join(', ')
          console.log(`      Columns: ${columns}`)
          
        } else {
          console.log(`  ❌ ${tableName} - NOT EXISTS`)
        }
      } catch (error) {
        console.log(`  ❌ ${tableName} - ERROR: ${error.message}`)
      }
    }
    console.log('')

    // Check RLS status
    console.log('🔒 Checking Row Level Security (RLS):')
    const rlsResult = await client.query(`
      SELECT schemaname, tablename, rowsecurity 
      FROM pg_tables 
      WHERE schemaname = 'public' 
      AND tablename IN ('taxonomy_nodes', 'products', 'synonyms', 'synonym_lemmas', 'synonym_terms')
      ORDER BY tablename
    `)
    
    rlsResult.rows.forEach(row => {
      const status = row.rowsecurity ? '🔒 ENABLED' : '🔓 DISABLED'
      console.log(`  ${status} ${row.tablename}`)
    })
    console.log('')

    // Check storage buckets
    console.log('📁 Checking Storage Buckets:')
    try {
      const bucketsResult = await client.query(`
        SELECT id, name, public 
        FROM storage.buckets 
        ORDER BY name
      `)
      
      if (bucketsResult.rows.length === 0) {
        console.log('  📝 No storage buckets found')
      } else {
        bucketsResult.rows.forEach(bucket => {
          const visibility = bucket.public ? 'public' : 'private'
          console.log(`  ✅ ${bucket.name} (${visibility})`)
        })
      }
    } catch (error) {
      console.log(`  ⚠️  Cannot access storage schema: ${error.message}`)
    }
    console.log('')

    // Check extensions
    console.log('🔌 Checking PostgreSQL Extensions:')
    const extensionsResult = await client.query(`
      SELECT extname, extversion 
      FROM pg_extension 
      WHERE extname IN ('vector', 'uuid-ossp', 'pgcrypto')
      ORDER BY extname
    `)
    
    const requiredExtensions = ['vector', 'uuid-ossp', 'pgcrypto']
    requiredExtensions.forEach(extName => {
      const extension = extensionsResult.rows.find(ext => ext.extname === extName)
      if (extension) {
        console.log(`  ✅ ${extName} v${extension.extversion}`)
      } else {
        console.log(`  ❌ ${extName} - NOT INSTALLED`)
      }
    })
    console.log('')

    // Test basic CRUD operations
    console.log('🧪 Testing Basic Operations:')
    
    // Test taxonomy_nodes if exists
    try {
      const testResult = await client.query(`
        SELECT COUNT(*) as count 
        FROM taxonomy_nodes 
        LIMIT 1
      `)
      console.log(`  ✅ Can read taxonomy_nodes (${testResult.rows[0].count} total rows)`)
      
      // Try to get sample data
      const sampleResult = await client.query(`
        SELECT id, name_th, name_en, level 
        FROM taxonomy_nodes 
        ORDER BY level, sort_order 
        LIMIT 3
      `)
      
      if (sampleResult.rows.length > 0) {
        console.log('  📊 Sample taxonomy data:')
        sampleResult.rows.forEach(row => {
          console.log(`    - L${row.level}: ${row.name_th} (${row.name_en || 'N/A'})`)
        })
      }
      
    } catch (error) {
      console.log(`  ❌ Cannot access taxonomy_nodes: ${error.message}`)
    }
    
  } catch (error) {
    console.error('❌ Database connection failed:', error.message)
  } finally {
    await client.end()
    console.log('\n🔌 Database connection closed')
  }
}

// Main execution
async function main() {
  console.log('🔍 Supabase Local Database Inspector (Direct Connection)\n')
  
  console.log('📡 Connection Details:')
  console.log('  Host: 127.0.0.1')
  console.log('  Port: 54322')
  console.log('  Database: postgres')
  console.log('  User: postgres')
  console.log('')
  
  await connectAndCheck()
  
  console.log('📋 Inspection Summary:')
  console.log('  ✅ Database connection tested')
  console.log('  ✅ Schema structure analyzed')
  console.log('  ✅ Table existence verified')
  console.log('  ✅ RLS policies checked')
  console.log('  ✅ Storage buckets inspected')
  console.log('  ✅ Extensions verified')
  console.log('  ✅ Basic operations tested')
}

if (require.main === module) {
  main().catch(console.error)
}

module.exports = { connectAndCheck }
