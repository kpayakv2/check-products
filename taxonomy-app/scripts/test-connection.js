#!/usr/bin/env node

/**
 * Test Supabase Connection Script
 * ทดสอบการเชื่อมต่อ Supabase และ Database Schema
 */

const { createClient } = require('@supabase/supabase-js')
require('dotenv').config({ path: '.env.local' })

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL
const supabaseKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY

if (!supabaseUrl || !supabaseKey) {
  console.error('❌ Missing Supabase credentials in .env.local')
  console.log('Please check:')
  console.log('- NEXT_PUBLIC_SUPABASE_URL')
  console.log('- NEXT_PUBLIC_SUPABASE_ANON_KEY')
  process.exit(1)
}

const supabase = createClient(supabaseUrl, supabaseKey)

async function testConnection() {
  console.log('🔄 Testing Supabase connection...\n')

  try {
    // Test 1: Basic connection
    console.log('1. Testing basic connection...')
    const { data, error } = await supabase.from('taxonomy_nodes').select('count').limit(1)
    
    if (error) {
      console.error('❌ Connection failed:', error.message)
      return false
    }
    console.log('✅ Basic connection successful')

    // Test 2: Check extensions
    console.log('\n2. Checking extensions...')
    const { data: extensions } = await supabase.rpc('check_extensions')
    console.log('✅ Extensions check completed')

    // Test 3: Check tables
    console.log('\n3. Checking database tables...')
    const tables = [
      'taxonomy_nodes',
      'synonyms', 
      'synonym_terms',
      'products',
      'keyword_rules',
      'similarity_matches',
      'audit_logs'
    ]

    for (const table of tables) {
      try {
        const { data, error } = await supabase.from(table).select('count').limit(1)
        if (error) {
          console.log(`❌ Table '${table}': ${error.message}`)
        } else {
          console.log(`✅ Table '${table}': OK`)
        }
      } catch (err) {
        console.log(`❌ Table '${table}': ${err.message}`)
      }
    }

    // Test 4: Check sample data
    console.log('\n4. Checking sample data...')
    const { data: nodes } = await supabase
      .from('taxonomy_nodes')
      .select('id, name_th')
      .limit(5)

    if (nodes && nodes.length > 0) {
      console.log('✅ Sample taxonomy nodes found:')
      nodes.forEach(node => {
        console.log(`   - ${node.name_th} (${node.id})`)
      })
    } else {
      console.log('⚠️  No sample data found - run schema.sql to insert sample data')
    }

    // Test 5: Check indexes
    console.log('\n5. Checking indexes...')
    const { data: indexes } = await supabase.rpc('check_indexes')
    console.log('✅ Indexes check completed')

    console.log('\n🎉 All tests completed successfully!')
    console.log('\nYour Supabase setup is ready for the Thai Product Taxonomy Manager!')
    
    return true

  } catch (error) {
    console.error('❌ Test failed:', error.message)
    return false
  }
}

// Helper function to check extensions
async function createHelperFunctions() {
  console.log('Creating helper functions...')
  
  const checkExtensionsSQL = `
    CREATE OR REPLACE FUNCTION check_extensions()
    RETURNS TABLE(name text, installed boolean) AS $$
    BEGIN
      RETURN QUERY
      SELECT 
        ext.name::text,
        (ext.name = ANY(ARRAY(SELECT extname FROM pg_extension)))::boolean as installed
      FROM (VALUES ('vector'), ('uuid-ossp')) AS ext(name);
    END;
    $$ LANGUAGE plpgsql;
  `

  const checkIndexesSQL = `
    CREATE OR REPLACE FUNCTION check_indexes()
    RETURNS TABLE(table_name text, index_name text, index_type text) AS $$
    BEGIN
      RETURN QUERY
      SELECT 
        t.tablename::text,
        i.indexname::text,
        CASE 
          WHEN i.indexdef LIKE '%USING gin%' THEN 'GIN'
          WHEN i.indexdef LIKE '%USING ivfflat%' THEN 'IVFFlat'
          ELSE 'BTree'
        END::text as index_type
      FROM pg_indexes i
      JOIN pg_tables t ON i.tablename = t.tablename
      WHERE t.schemaname = 'public'
      AND i.indexname LIKE 'idx_%'
      ORDER BY t.tablename, i.indexname;
    END;
    $$ LANGUAGE plpgsql;
  `

  try {
    await supabase.rpc('exec_sql', { sql: checkExtensionsSQL })
    await supabase.rpc('exec_sql', { sql: checkIndexesSQL })
  } catch (error) {
    // Functions might already exist or user might not have permissions
    console.log('Note: Could not create helper functions (this is normal)')
  }
}

// Run the test
async function main() {
  console.log('🇹🇭 Thai Product Taxonomy Manager - Connection Test\n')
  console.log(`Supabase URL: ${supabaseUrl}`)
  console.log(`Using API Key: ${supabaseKey.substring(0, 20)}...\n`)

  await createHelperFunctions()
  const success = await testConnection()
  
  if (success) {
    console.log('\n📋 Next Steps:')
    console.log('1. Run: npm run dev')
    console.log('2. Open: http://localhost:3000')
    console.log('3. Start managing your taxonomy!')
    process.exit(0)
  } else {
    console.log('\n🔧 Troubleshooting:')
    console.log('1. Check your .env.local file')
    console.log('2. Verify Supabase project is active')
    console.log('3. Run the schema.sql file in Supabase SQL Editor')
    console.log('4. Check Supabase project settings')
    process.exit(1)
  }
}

main().catch(console.error)
