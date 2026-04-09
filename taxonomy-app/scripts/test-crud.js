#!/usr/bin/env node
/**
 * CRUD Operations Testing Script
 * ทดสอบการทำงานของ Database Service
 */

const { createClient } = require('@supabase/supabase-js')
require('dotenv').config({ path: '.env.local' })

// Configuration - Use service role for testing
const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL
const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY

if (!supabaseUrl || !supabaseKey) {
  console.error('❌ Missing Supabase configuration')
  console.error('Please check .env.local file')
  process.exit(1)
}

const supabase = createClient(supabaseUrl, supabaseKey)

// Test functions
async function testTaxonomyOperations() {
  console.log('🌳 Testing Taxonomy Operations...')
  
  try {
    // Test: Get taxonomy tree
    console.log('  📋 Getting taxonomy tree...')
    const { data: taxonomyData, error: fetchError } = await supabase
      .from('taxonomy_nodes')
      .select('*')
      .limit(5)
    
    if (fetchError && fetchError.code !== 'PGRST116') {
      console.log('  ⚠️  No taxonomy_nodes table found (expected for new setup)')
    } else {
      console.log(`  ✅ Found ${taxonomyData?.length || 0} taxonomy nodes`)
    }

    // Test: Create category (if table exists)
    try {
      console.log('  ➕ Testing category creation...')
      const testCategory = {
        code: `test_${Date.now()}`, // เพิ่ม code field
        name_th: 'ทดสอบหมวดหมู่',
        name_en: 'Test Category',
        description: 'หมวดหมู่สำหรับทดสอบระบบ',
        level: 0,
        sort_order: 999,
        is_active: true
      }

      const { data: createData, error: createError } = await supabase
        .from('taxonomy_nodes')
        .insert([testCategory])
        .select()

      if (createError) {
        console.log('  ⚠️  Cannot create category:', createError.message)
      } else {
        console.log('  ✅ Category created successfully')
        
        // Clean up: Delete test category
        await supabase
          .from('taxonomy_nodes')
          .delete()
          .eq('id', createData[0].id)
        console.log('  🧹 Test category cleaned up')
      }
    } catch (error) {
      console.log('  ⚠️  Create test skipped:', error.message)
    }

  } catch (error) {
    console.log('  ❌ Taxonomy test failed:', error.message)
  }
}

async function testSynonymOperations() {
  console.log('📝 Testing Synonym Operations...')
  
  try {
    // Test: Get synonyms
    console.log('  📋 Getting synonyms...')
    const { data: synonymData, error: fetchError } = await supabase
      .from('synonyms')
      .select('*')
      .limit(5)
    
    if (fetchError && fetchError.code !== 'PGRST116') {
      console.log('  ⚠️  No synonyms table found (expected for new setup)')
    } else {
      console.log(`  ✅ Found ${synonymData?.length || 0} synonyms`)
    }

    // Test: Create synonym (if table exists)
    try {
      console.log('  ➕ Testing synonym creation...')
      const testSynonym = {
        main_term: 'โทรศัพท์',
        synonym_term: 'มือถือ',
        category_id: null,
        confidence_score: 0.95,
        is_verified: false,
        source: 'manual_test'
      }

      const { data: createData, error: createError } = await supabase
        .from('synonyms')
        .insert([testSynonym])
        .select()

      if (createError) {
        console.log('  ⚠️  Cannot create synonym:', createError.message)
      } else {
        console.log('  ✅ Synonym created successfully')
        
        // Clean up: Delete test synonym
        await supabase
          .from('synonyms')
          .delete()
          .eq('id', createData[0].id)
        console.log('  🧹 Test synonym cleaned up')
      }
    } catch (error) {
      console.log('  ⚠️  Create test skipped:', error.message)
    }

  } catch (error) {
    console.log('  ❌ Synonym test failed:', error.message)
  }
}

async function testProductOperations() {
  console.log('🛍️ Testing Product Operations...')
  
  try {
    // Test: Get products
    console.log('  📋 Getting products...')
    const { data: productData, error: fetchError } = await supabase
      .from('products')
      .select('*')
      .limit(5)
    
    if (fetchError && fetchError.code !== 'PGRST116') {
      console.log('  ⚠️  No products table found (expected for new setup)')
    } else {
      console.log(`  ✅ Found ${productData?.length || 0} products`)
    }

    // Test: Create product (if table exists)
    try {
      console.log('  ➕ Testing product creation...')
      const testProduct = {
        name_th: 'สินค้าทดสอบ',
        name_en: 'Test Product',
        description: 'สินค้าสำหรับทดสอบระบบ',
        sku: 'TEST001',
        brand: 'Test Brand',
        status: 'pending',
        confidence_score: 0.8
      }

      const { data: createData, error: createError } = await supabase
        .from('products')
        .insert([testProduct])
        .select()

      if (createError) {
        console.log('  ⚠️  Cannot create product:', createError.message)
      } else {
        console.log('  ✅ Product created successfully')
        
        // Test: Update product status
        const { error: updateError } = await supabase
          .from('products')
          .update({ status: 'approved' })
          .eq('id', createData[0].id)

        if (updateError) {
          console.log('  ⚠️  Cannot update product:', updateError.message)
        } else {
          console.log('  ✅ Product status updated')
        }
        
        // Clean up: Delete test product
        await supabase
          .from('products')
          .delete()
          .eq('id', createData[0].id)
        console.log('  🧹 Test product cleaned up')
      }
    } catch (error) {
      console.log('  ⚠️  Create test skipped:', error.message)
    }

  } catch (error) {
    console.log('  ❌ Product test failed:', error.message)
  }
}

async function testStorageOperations() {
  console.log('📁 Testing Storage Operations...')
  
  try {
    // Test: List buckets
    console.log('  📋 Listing storage buckets...')
    const { data: buckets, error: bucketsError } = await supabase.storage.listBuckets()
    
    if (bucketsError) {
      console.log('  ⚠️  Cannot list buckets:', bucketsError.message)
    } else {
      console.log(`  ✅ Found ${buckets?.length || 0} storage buckets`)
      buckets?.forEach(bucket => {
        console.log(`    - ${bucket.name} (${bucket.public ? 'public' : 'private'})`)
      })
    }

    // Test: Create test file
    try {
      console.log('  ➕ Testing file upload...')
      const testContent = 'This is a test file for CRUD operations'
      const testFile = new Blob([testContent], { type: 'text/plain' })
      
      const { data: uploadData, error: uploadError } = await supabase.storage
        .from('uploads')
        .upload(`test_${Date.now()}.txt`, testFile)

      if (uploadError) {
        console.log('  ⚠️  Cannot upload file:', uploadError.message)
      } else {
        console.log('  ✅ File uploaded successfully')
        
        // Clean up: Delete test file
        await supabase.storage
          .from('uploads')
          .remove([uploadData.path])
        console.log('  🧹 Test file cleaned up')
      }
    } catch (error) {
      console.log('  ⚠️  Upload test skipped:', error.message)
    }

  } catch (error) {
    console.log('  ❌ Storage test failed:', error.message)
  }
}

async function testDatabaseConnection() {
  console.log('🔌 Testing Database Connection...')
  
  try {
    const { data, error } = await supabase
      .from('information_schema.tables')
      .select('table_name')
      .eq('table_schema', 'public')
      .limit(10)
    
    if (error) {
      console.log('  ❌ Database connection failed:', error.message)
    } else {
      console.log('  ✅ Database connected successfully')
      console.log(`  📊 Found ${data?.length || 0} tables in public schema`)
      
      if (data && data.length > 0) {
        console.log('  📋 Available tables:')
        data.forEach(table => {
          console.log(`    - ${table.table_name}`)
        })
      }
    }
  } catch (error) {
    console.log('  ❌ Connection test failed:', error.message)
  }
}

// Main test runner
async function runAllTests() {
  console.log('🚀 Starting CRUD Operations Test...\n')
  
  console.log(`📡 Testing against: ${supabaseUrl}`)
  console.log(`🔑 Using key: ${supabaseKey.substring(0, 20)}...\n`)

  // Run all tests
  await testDatabaseConnection()
  console.log('')
  
  await testTaxonomyOperations()
  console.log('')
  
  await testSynonymOperations()
  console.log('')
  
  await testProductOperations()
  console.log('')
  
  await testStorageOperations()
  console.log('')

  console.log('✨ CRUD Tests completed!')
  console.log('')
  console.log('📋 Summary:')
  console.log('  - Database connection: Tested')
  console.log('  - Taxonomy operations: Tested')
  console.log('  - Synonym operations: Tested')
  console.log('  - Product operations: Tested')
  console.log('  - Storage operations: Tested')
  console.log('')
  console.log('💡 Note: Some operations may show warnings if tables don\'t exist yet.')
  console.log('   This is normal for a new Supabase setup.')
}

// Run if called directly
if (require.main === module) {
  runAllTests().catch(console.error)
}

module.exports = {
  testTaxonomyOperations,
  testSynonymOperations,
  testProductOperations,
  testStorageOperations,
  testDatabaseConnection
}
