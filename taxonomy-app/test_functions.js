const SUPABASE_URL = 'http://127.0.0.1:54321'
const SUPABASE_ANON_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZS1kZW1vIiwicm9sZSI6ImFub24iLCJleHAiOjE5ODM4MTI5OTZ9.CRXP1A7WOeoJeXxjNni43kdQwgnWNReilDMblYTn_I0'

// Test category-suggestions function
async function testCategorySuggestions() {
  console.log('🧪 Testing category-suggestions function...')
  
  const response = await fetch(`${SUPABASE_URL}/functions/v1/category-suggestions`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${SUPABASE_ANON_KEY}`
    },
    body: JSON.stringify({
      text: 'iPhone 15 Pro Max 256GB สีดำ',
      options: {
        maxSuggestions: 5,
        minConfidence: 0.3,
        includeExplanation: true
      }
    })
  })
  
  if (response.ok) {
    const result = await response.json()
    console.log('✅ category-suggestions:', result)
  } else {
    const error = await response.text()
    console.log('❌ category-suggestions error:', error)
  }
}

// Test generate-embeddings function
async function testGenerateEmbeddings() {
  console.log('🧪 Testing generate-embeddings function...')
  
  const response = await fetch(`${SUPABASE_URL}/functions/v1/generate-embeddings`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${SUPABASE_ANON_KEY}`
    },
    body: JSON.stringify({
      texts: ['iPhone 15', 'Samsung Galaxy S24'],
      model: 'text-embedding-ada-002'
    })
  })
  
  if (response.ok) {
    const result = await response.json()
    console.log('✅ generate-embeddings:', result)
  } else {
    const error = await response.text()
    console.log('❌ generate-embeddings error:', error)
  }
}

// Test hybrid-search function
async function testHybridSearch() {
  console.log('🧪 Testing hybrid-search function...')
  
  const response = await fetch(`${SUPABASE_URL}/functions/v1/hybrid-search`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${SUPABASE_ANON_KEY}`
    },
    body: JSON.stringify({
      query: 'smartphone',
      type: 'text',
      limit: 5
    })
  })
  
  if (response.ok) {
    const result = await response.json()
    console.log('✅ hybrid-search:', result)
  } else {
    const error = await response.text()
    console.log('❌ hybrid-search error:', error)
  }
}

// Test exec-sql function
async function testExecSQL() {
  console.log('🧪 Testing exec-sql function...')
  
  const response = await fetch(`${SUPABASE_URL}/functions/v1/exec-sql`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${SUPABASE_ANON_KEY}`
    },
    body: JSON.stringify({
      query: 'SELECT 1 as test_value',
      params: []
    })
  })
  
  if (response.ok) {
    const result = await response.json()
    console.log('✅ exec-sql:', result)
  } else {
    const error = await response.text()
    console.log('❌ exec-sql error:', error)
  }
}

// Run all tests
async function runTests() {
  console.log('🚀 Starting Supabase Functions Tests...')
  console.log('🌐 Testing connection to:', SUPABASE_URL)
  console.log('=' .repeat(50))
  
  try {
    await testCategorySuggestions()
    console.log('')
    
    await testGenerateEmbeddings()
    console.log('')
    
    await testHybridSearch()
    console.log('')
    
    await testExecSQL()
    console.log('')
    
    console.log('🎉 All tests completed!')
  } catch (error) {
    console.error('💥 Test execution failed:', error)
  }
}

// Run tests
runTests()