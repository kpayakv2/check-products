#!/usr/bin/env node
/**
 * API Testing Script for Product Deduplication System
 * ทดสอบ API endpoints และ Edge Functions
 */

const https = require('https');
const fs = require('fs');
const path = require('path');

// Configuration
const config = {
  supabaseUrl: process.env.NEXT_PUBLIC_SUPABASE_URL || 'your_supabase_url',
  supabaseKey: process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY || 'your_anon_key',
  testTimeout: 10000
};

// Test data
const testData = {
  categoryTest: {
    text: "iPhone 15 Pro Max 256GB",
    options: {
      maxSuggestions: 3,
      minConfidence: 0.3,
      includeExplanation: true
    }
  },
  embeddingTest: {
    texts: ["iPhone 15", "Samsung Galaxy S24"],
    model: "text-embedding-ada-002"
  }
};

// Utility functions
function makeRequest(url, options, data = null) {
  return new Promise((resolve, reject) => {
    const req = https.request(url, options, (res) => {
      let body = '';
      res.on('data', (chunk) => body += chunk);
      res.on('end', () => {
        try {
          const result = JSON.parse(body);
          resolve({ status: res.statusCode, data: result });
        } catch (e) {
          resolve({ status: res.statusCode, data: body });
        }
      });
    });

    req.on('error', reject);
    req.setTimeout(config.testTimeout, () => {
      req.destroy();
      reject(new Error('Request timeout'));
    });

    if (data) {
      req.write(JSON.stringify(data));
    }
    req.end();
  });
}

// Test functions
async function testCategorySuggestions() {
  console.log('🧪 Testing Category Suggestions API...');
  
  try {
    const url = `${config.supabaseUrl}/functions/v1/category-suggestions`;
    const options = {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${config.supabaseKey}`
      }
    };

    const result = await makeRequest(url, options, testData.categoryTest);
    
    if (result.status === 200) {
      console.log('✅ Category Suggestions API: OK');
      console.log(`   - Suggestions: ${result.data.suggestions?.length || 0}`);
      console.log(`   - Processing time: ${result.data.processingTime}ms`);
    } else {
      console.log('❌ Category Suggestions API: FAILED');
      console.log(`   - Status: ${result.status}`);
      console.log(`   - Error: ${result.data.error || 'Unknown error'}`);
    }
  } catch (error) {
    console.log('❌ Category Suggestions API: ERROR');
    console.log(`   - ${error.message}`);
  }
}

async function testEmbeddingGeneration() {
  console.log('🧪 Testing Embedding Generation API...');
  
  try {
    const url = `${config.supabaseUrl}/functions/v1/generate-embeddings`;
    const options = {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${config.supabaseKey}`
      }
    };

    const result = await makeRequest(url, options, testData.embeddingTest);
    
    if (result.status === 200) {
      console.log('✅ Embedding Generation API: OK');
      console.log(`   - Embeddings: ${result.data.embeddings?.length || 0}`);
      console.log(`   - Model: ${result.data.model}`);
    } else {
      console.log('❌ Embedding Generation API: FAILED');
      console.log(`   - Status: ${result.status}`);
      console.log(`   - Error: ${result.data.error || 'Unknown error'}`);
    }
  } catch (error) {
    console.log('❌ Embedding Generation API: ERROR');
    console.log(`   - ${error.message}`);
  }
}

async function testHybridSearch() {
  console.log('🧪 Testing Hybrid Search API...');
  
  try {
    const url = `${config.supabaseUrl}/functions/v1/hybrid-search`;
    const options = {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${config.supabaseKey}`
      }
    };

    const searchData = {
      query: "smartphone Apple",
      type: "hybrid",
      limit: 5
    };

    const result = await makeRequest(url, options, searchData);
    
    if (result.status === 200) {
      console.log('✅ Hybrid Search API: OK');
      console.log(`   - Results: ${result.data.results?.length || 0}`);
      console.log(`   - Search time: ${result.data.searchTime}ms`);
    } else {
      console.log('❌ Hybrid Search API: FAILED');
      console.log(`   - Status: ${result.status}`);
      console.log(`   - Error: ${result.data.error || 'Unknown error'}`);
    }
  } catch (error) {
    console.log('❌ Hybrid Search API: ERROR');
    console.log(`   - ${error.message}`);
  }
}

async function testProductDeduplication() {
  console.log('🧪 Testing Product Deduplication API...');
  
  try {
    const url = `${config.supabaseUrl}/functions/v1/product-deduplication`;
    const options = {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${config.supabaseKey}`
      }
    };

    // Note: This would need actual file paths in Supabase Storage
    const deduplicationData = {
      oldProductsPath: "test/old_products.csv",
      newProductsPath: "test/new_products.csv",
      threshold: 0.75
    };

    const result = await makeRequest(url, options, deduplicationData);
    
    if (result.status === 200) {
      console.log('✅ Product Deduplication API: OK');
      console.log(`   - Unique products: ${result.data.stats?.autoApproved || 0}`);
      console.log(`   - Need review: ${result.data.stats?.needsReview || 0}`);
    } else {
      console.log('❌ Product Deduplication API: FAILED');
      console.log(`   - Status: ${result.status}`);
      console.log(`   - Error: ${result.data.error || 'Unknown error'}`);
    }
  } catch (error) {
    console.log('❌ Product Deduplication API: ERROR');
    console.log(`   - ${error.message}`);
  }
}

// Main test runner
async function runTests() {
  console.log('🚀 Starting API Tests...\n');
  
  // Check configuration
  if (!config.supabaseUrl || config.supabaseUrl === 'your_supabase_url') {
    console.log('❌ Missing Supabase configuration');
    console.log('   Please set NEXT_PUBLIC_SUPABASE_URL environment variable');
    process.exit(1);
  }

  console.log(`📡 Testing against: ${config.supabaseUrl}`);
  console.log(`🔑 Using key: ${config.supabaseKey.substring(0, 20)}...\n`);

  // Run tests
  await testCategorySuggestions();
  await testEmbeddingGeneration();
  await testHybridSearch();
  await testProductDeduplication();

  console.log('\n✨ API Tests completed!');
}

// Run if called directly
if (require.main === module) {
  runTests().catch(console.error);
}

module.exports = {
  testCategorySuggestions,
  testEmbeddingGeneration,
  testHybridSearch,
  testProductDeduplication
};
