import { createClient } from '@supabase/supabase-js'

// Test Database Configuration - ใช้ค่าจาก .env.local เหมือนกับระบบหลัก
const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!
const supabaseKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!

if (!supabaseUrl || !supabaseKey) {
  throw new Error('❌ Missing Supabase credentials in .env.local')
}

console.log('🔗 Connecting to Supabase:', supabaseUrl)

export const testSupabase = createClient(supabaseUrl, supabaseKey)

// Test Data Seeds
export const seedTaxonomyData = [
  {
    id: 'test-tax-1',
    name: 'Test Electronics',
    code: 'TEST001',
    parent_id: null,
    level: 1,
    sort_order: 1,
    is_active: true,
  },
  {
    id: 'test-tax-2', 
    name: 'Test Smartphones',
    code: 'TEST002',
    parent_id: 'test-tax-1',
    level: 2,
    sort_order: 1,
    is_active: true,
  }
]

export const seedSynonymData = [
  {
    id: 'test-syn-1',
    lemma: 'test-smartphone',
    category_id: 'test-tax-2',
    confidence_score: 0.95,
    is_verified: true,
  }
]

export const seedSynonymTermData = [
  {
    id: 'test-term-1',
    synonym_id: 'test-syn-1',
    term: 'test smart phone',
    language: 'en',
  },
  {
    id: 'test-term-2',
    synonym_id: 'test-syn-1',
    term: 'เทสมือถือ',
    language: 'th',
  }
]

export const seedProductData = [
  {
    id: 'test-prod-1',
    name: 'Test iPhone 15 Pro',
    description: 'Test latest iPhone model',
    category_id: 'test-tax-2',
    status: 'pending',
    similarity_score: 0.85,
  }
]

// Database Setup Functions
export async function setupTestDatabase() {
  console.log('🔄 Setting up test database...')
  
  try {
    // Clear existing test data
    await clearTestData()
    
    // Insert seed data
    await insertSeedData()
    
    console.log('✅ Test database setup complete')
  } catch (error) {
    console.error('❌ Test database setup failed:', error)
    throw error
  }
}

export async function clearTestData() {
  console.log('🧹 Clearing test data...')
  
  // Delete in correct order (foreign key constraints)
  await testSupabase.from('products').delete().like('id', 'test-%')
  await testSupabase.from('synonym_terms').delete().like('id', 'test-%')
  await testSupabase.from('synonyms').delete().like('id', 'test-%')
  await testSupabase.from('taxonomy_nodes').delete().like('id', 'test-%')
}

export async function insertSeedData() {
  console.log('🌱 Inserting seed data...')
  
  // Insert taxonomy nodes
  const { error: taxError } = await testSupabase
    .from('taxonomy_nodes')
    .insert(seedTaxonomyData)
  
  if (taxError) throw taxError
  
  // Insert synonyms
  const { error: synError } = await testSupabase
    .from('synonyms')
    .insert(seedSynonymData)
  
  if (synError) throw synError
  
  // Insert synonym terms
  const { error: termError } = await testSupabase
    .from('synonym_terms')
    .insert(seedSynonymTermData)
  
  if (termError) throw termError
  
  // Insert products
  const { error: prodError } = await testSupabase
    .from('products')
    .insert(seedProductData)
  
  if (prodError) throw prodError
}

// Test Utilities
export async function getTestTaxonomyNodes() {
  const { data, error } = await testSupabase
    .from('taxonomy_nodes')
    .select('*')
    .like('id', 'test-%')
    .order('level', { ascending: true })
  
  if (error) throw error
  return data
}

export async function getTestSynonyms() {
  const { data, error } = await testSupabase
    .from('synonyms')
    .select(`
      *,
      synonym_terms (*)
    `)
    .like('id', 'test-%')
  
  if (error) throw error
  return data
}

export async function getTestProducts() {
  const { data, error } = await testSupabase
    .from('products')
    .select('*')
    .like('id', 'test-%')
  
  if (error) throw error
  return data
}
