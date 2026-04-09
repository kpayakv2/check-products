#!/usr/bin/env node

/**
 * Test Setup Script
 * Sets up test environment and dependencies
 */

const { execSync } = require('child_process')
const fs = require('fs')
const path = require('path')

console.log('🧪 Setting up test environment...\n')

// Check if we're in the right directory
if (!fs.existsSync('package.json')) {
  console.error('❌ Error: package.json not found. Please run this script from the project root.')
  process.exit(1)
}

// Install test dependencies if not already installed
console.log('📦 Installing test dependencies...')
try {
  execSync('npm install --save-dev @playwright/test @testing-library/jest-dom @testing-library/react @testing-library/user-event @types/jest jest jest-environment-jsdom msw supertest @types/supertest', {
    stdio: 'inherit'
  })
  console.log('✅ Test dependencies installed successfully\n')
} catch (error) {
  console.error('❌ Failed to install test dependencies:', error.message)
  process.exit(1)
}

// Install Playwright browsers
console.log('🎭 Installing Playwright browsers...')
try {
  execSync('npx playwright install', { stdio: 'inherit' })
  console.log('✅ Playwright browsers installed successfully\n')
} catch (error) {
  console.error('❌ Failed to install Playwright browsers:', error.message)
  console.log('⚠️  You can install them later with: npx playwright install\n')
}

// Create test directories if they don't exist
const testDirs = [
  '__tests__',
  '__tests__/utils',
  '__tests__/components',
  '__tests__/components/Taxonomy',
  '__tests__/components/Synonym',
  '__tests__/components/Product',
  '__tests__/api',
  '__tests__/api/taxonomy',
  '__tests__/api/synonyms',
  '__tests__/api/products',
  'e2e',
  '__mocks__'
]

console.log('📁 Creating test directories...')
testDirs.forEach(dir => {
  const dirPath = path.join(process.cwd(), dir)
  if (!fs.existsSync(dirPath)) {
    fs.mkdirSync(dirPath, { recursive: true })
    console.log(`  ✅ Created ${dir}/`)
  } else {
    console.log(`  ⏭️  ${dir}/ already exists`)
  }
})

// Check if test configuration files exist
const configFiles = [
  'jest.config.js',
  'jest.setup.js',
  'playwright.config.ts'
]

console.log('\n🔧 Checking test configuration files...')
configFiles.forEach(file => {
  const filePath = path.join(process.cwd(), file)
  if (fs.existsSync(filePath)) {
    console.log(`  ✅ ${file} exists`)
  } else {
    console.log(`  ⚠️  ${file} not found - please create it`)
  }
})

// Verify package.json scripts
console.log('\n📜 Checking package.json test scripts...')
const packageJson = JSON.parse(fs.readFileSync('package.json', 'utf8'))
const requiredScripts = {
  'test': 'jest',
  'test:watch': 'jest --watch',
  'test:coverage': 'jest --coverage',
  'test:ci': 'jest --ci --coverage --watchAll=false',
  'test:e2e': 'playwright test',
  'test:e2e:ui': 'playwright test --ui'
}

const missingScripts = []
Object.entries(requiredScripts).forEach(([script, command]) => {
  if (packageJson.scripts && packageJson.scripts[script]) {
    console.log(`  ✅ ${script}: ${packageJson.scripts[script]}`)
  } else {
    console.log(`  ⚠️  ${script} script missing`)
    missingScripts.push({ script, command })
  }
})

if (missingScripts.length > 0) {
  console.log('\n📝 Missing scripts that should be added to package.json:')
  missingScripts.forEach(({ script, command }) => {
    console.log(`  "${script}": "${command}"`)
  })
}

// Check environment variables
console.log('\n🌍 Checking test environment variables...')
const envFile = '.env.test'
if (fs.existsSync(envFile)) {
  console.log(`  ✅ ${envFile} exists`)
} else {
  console.log(`  ⚠️  ${envFile} not found - consider creating it for test-specific environment variables`)
}

// Run a quick test to verify setup
console.log('\n🧪 Running setup verification...')
try {
  execSync('npm run test -- --passWithNoTests --verbose=false', { stdio: 'pipe' })
  console.log('  ✅ Jest is working correctly')
} catch (error) {
  console.log('  ⚠️  Jest setup may need attention')
  console.log('  Error:', error.message.split('\n')[0])
}

// Check if Playwright is working
try {
  execSync('npx playwright --version', { stdio: 'pipe' })
  console.log('  ✅ Playwright is working correctly')
} catch (error) {
  console.log('  ⚠️  Playwright setup may need attention')
}

console.log('\n🎉 Test setup complete!')
console.log('\n📋 Next steps:')
console.log('  1. Review and update test configuration files if needed')
console.log('  2. Add missing package.json scripts if any')
console.log('  3. Create .env.test file for test environment variables')
console.log('  4. Run tests to verify everything works:')
console.log('     - npm run test (unit tests)')
console.log('     - npm run test:e2e (E2E tests)')
console.log('  5. Check test coverage: npm run test:coverage')
console.log('\n📖 For more information, see TESTING.md')

console.log('\n🚀 Happy testing!')
