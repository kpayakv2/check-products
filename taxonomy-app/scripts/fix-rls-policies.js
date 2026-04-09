#!/usr/bin/env node
/**
 * Fix RLS Policies Script
 * แก้ไข RLS policies ให้ใช้ auth.uid() แทน auth.role()
 */

const { Client } = require('pg')
require('dotenv').config({ path: '.env.local' })

const client = new Client({
  host: '127.0.0.1',
  port: 54322,
  database: 'postgres',
  user: 'postgres',
  password: 'postgres'
})

async function fixRLSPolicies() {
  try {
    console.log('🔒 Fixing RLS Policies...\n')
    await client.connect()

    // Tables to fix
    const tables = ['taxonomy_nodes', 'products', 'synonym_lemmas', 'synonym_terms']

    for (const tableName of tables) {
      console.log(`🔧 Fixing policies for ${tableName}...`)
      
      // Drop existing policies
      await client.query(`DROP POLICY IF EXISTS "${tableName}_read" ON ${tableName}`)
      await client.query(`DROP POLICY IF EXISTS "${tableName}_insert" ON ${tableName}`)
      await client.query(`DROP POLICY IF EXISTS "${tableName}_update" ON ${tableName}`)
      await client.query(`DROP POLICY IF EXISTS "${tableName}_delete" ON ${tableName}`)
      
      // Create new policies using auth.uid()
      await client.query(`
        CREATE POLICY "Allow authenticated read" ON ${tableName} 
        FOR SELECT USING (auth.uid() IS NOT NULL)
      `)
      
      await client.query(`
        CREATE POLICY "Allow authenticated insert" ON ${tableName} 
        FOR INSERT WITH CHECK (auth.uid() IS NOT NULL)
      `)
      
      await client.query(`
        CREATE POLICY "Allow authenticated update" ON ${tableName} 
        FOR UPDATE USING (auth.uid() IS NOT NULL)
      `)
      
      await client.query(`
        CREATE POLICY "Allow authenticated delete" ON ${tableName} 
        FOR DELETE USING (auth.uid() IS NOT NULL)
      `)
      
      console.log(`  ✅ ${tableName} policies updated`)
    }

    // Create storage bucket if not exists
    console.log('\n📁 Creating storage bucket...')
    try {
      await client.query(`
        INSERT INTO storage.buckets (id, name, public) 
        VALUES ('uploads', 'uploads', false)
        ON CONFLICT (id) DO NOTHING
      `)
      console.log('  ✅ uploads bucket created/verified')
      
      // Create storage policy
      await client.query(`
        DROP POLICY IF EXISTS "Allow authenticated uploads" ON storage.objects
      `)
      
      await client.query(`
        CREATE POLICY "Allow authenticated uploads" ON storage.objects 
        FOR ALL USING (bucket_id = 'uploads' AND auth.uid() IS NOT NULL)
      `)
      console.log('  ✅ storage policy created')
      
    } catch (storageError) {
      console.log(`  ⚠️  Storage setup: ${storageError.message}`)
    }

    console.log('\n✅ RLS Policies fixed successfully!')
    
  } catch (error) {
    console.error('❌ Error fixing RLS policies:', error.message)
  } finally {
    await client.end()
  }
}

if (require.main === module) {
  fixRLSPolicies().catch(console.error)
}

module.exports = { fixRLSPolicies }
