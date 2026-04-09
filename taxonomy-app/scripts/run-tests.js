#!/usr/bin/env node

/**
 * Test Runner Script
 * Provides convenient commands for running different types of tests
 */

const { execSync, spawn } = require('child_process')
const fs = require('fs')
const path = require('path')

// Parse command line arguments
const args = process.argv.slice(2)
const command = args[0]
const options = args.slice(1)

// Available commands
const commands = {
  unit: 'Run unit tests',
  api: 'Run API tests',
  e2e: 'Run E2E tests',
  coverage: 'Run tests with coverage',
  watch: 'Run tests in watch mode',
  ci: 'Run tests in CI mode',
  all: 'Run all tests',
  help: 'Show this help message'
}

function showHelp() {
  console.log('🧪 Test Runner\n')
  console.log('Usage: node scripts/run-tests.js <command> [options]\n')
  console.log('Available commands:')
  Object.entries(commands).forEach(([cmd, desc]) => {
    console.log(`  ${cmd.padEnd(10)} ${desc}`)
  })
  console.log('\nExamples:')
  console.log('  node scripts/run-tests.js unit')
  console.log('  node scripts/run-tests.js e2e --headed')
  console.log('  node scripts/run-tests.js coverage')
  console.log('  node scripts/run-tests.js watch --testNamePattern="TaxonomyTree"')
}

function runCommand(cmd, args = [], options = {}) {
  console.log(`🚀 Running: ${cmd} ${args.join(' ')}\n`)
  
  try {
    const result = execSync(`${cmd} ${args.join(' ')}`, {
      stdio: 'inherit',
      cwd: process.cwd(),
      ...options
    })
    return true
  } catch (error) {
    console.error(`❌ Command failed: ${error.message}`)
    return false
  }
}

function runCommandAsync(cmd, args = []) {
  console.log(`🚀 Running: ${cmd} ${args.join(' ')}\n`)
  
  const child = spawn(cmd, args, {
    stdio: 'inherit',
    cwd: process.cwd(),
    shell: true
  })
  
  return new Promise((resolve, reject) => {
    child.on('close', (code) => {
      if (code === 0) {
        resolve(true)
      } else {
        reject(new Error(`Command failed with code ${code}`))
      }
    })
    
    child.on('error', (error) => {
      reject(error)
    })
  })
}

async function runTests() {
  // Check if we're in the right directory
  if (!fs.existsSync('package.json')) {
    console.error('❌ Error: package.json not found. Please run this script from the project root.')
    process.exit(1)
  }

  switch (command) {
    case 'unit':
      console.log('🧪 Running unit tests...')
      return runCommand('npm', ['run', 'test', '--', '--testPathIgnorePatterns=e2e', ...options])

    case 'api':
      console.log('🔌 Running API tests...')
      return runCommand('npm', ['run', 'test', '--', '--testPathPattern=api', ...options])

    case 'e2e':
      console.log('🎭 Running E2E tests...')
      // Check if Playwright is installed
      try {
        execSync('npx playwright --version', { stdio: 'pipe' })
      } catch (error) {
        console.log('⚠️  Playwright not found. Installing...')
        runCommand('npx', ['playwright', 'install'])
      }
      return runCommand('npm', ['run', 'test:e2e', ...options])

    case 'coverage':
      console.log('📊 Running tests with coverage...')
      const success = runCommand('npm', ['run', 'test:coverage', ...options])
      if (success) {
        console.log('\n📈 Coverage report generated!')
        console.log('Open coverage/lcov-report/index.html to view the report')
      }
      return success

    case 'watch':
      console.log('👀 Running tests in watch mode...')
      return runCommand('npm', ['run', 'test:watch', ...options])

    case 'ci':
      console.log('🤖 Running tests in CI mode...')
      return runCommand('npm', ['run', 'test:ci', ...options])

    case 'all':
      console.log('🎯 Running all tests...')
      
      console.log('\n1️⃣ Running unit tests...')
      const unitSuccess = runCommand('npm', ['run', 'test', '--', '--testPathIgnorePatterns=e2e'])
      
      if (!unitSuccess) {
        console.error('❌ Unit tests failed. Stopping.')
        return false
      }
      
      console.log('\n2️⃣ Running E2E tests...')
      const e2eSuccess = runCommand('npm', ['run', 'test:e2e'])
      
      if (!e2eSuccess) {
        console.error('❌ E2E tests failed.')
        return false
      }
      
      console.log('\n✅ All tests passed!')
      return true

    case 'help':
    case undefined:
      showHelp()
      return true

    default:
      console.error(`❌ Unknown command: ${command}`)
      showHelp()
      return false
  }
}

// Additional utility functions
function checkTestEnvironment() {
  console.log('🔍 Checking test environment...\n')
  
  const checks = [
    {
      name: 'Jest configuration',
      check: () => fs.existsSync('jest.config.js') || fs.existsSync('jest.config.ts'),
      message: 'jest.config.js or jest.config.ts'
    },
    {
      name: 'Jest setup file',
      check: () => fs.existsSync('jest.setup.js') || fs.existsSync('jest.setup.ts'),
      message: 'jest.setup.js or jest.setup.ts'
    },
    {
      name: 'Playwright configuration',
      check: () => fs.existsSync('playwright.config.ts') || fs.existsSync('playwright.config.js'),
      message: 'playwright.config.ts or playwright.config.js'
    },
    {
      name: 'Test directories',
      check: () => fs.existsSync('__tests__') && fs.existsSync('e2e'),
      message: '__tests__ and e2e directories'
    }
  ]
  
  let allGood = true
  checks.forEach(({ name, check, message }) => {
    if (check()) {
      console.log(`✅ ${name}`)
    } else {
      console.log(`❌ ${name} - Missing: ${message}`)
      allGood = false
    }
  })
  
  if (allGood) {
    console.log('\n🎉 Test environment looks good!')
  } else {
    console.log('\n⚠️  Some issues found. Run: node scripts/test-setup.js')
  }
  
  return allGood
}

// Handle special commands
if (command === 'check') {
  checkTestEnvironment()
  process.exit(0)
}

// Run the main function
runTests()
  .then((success) => {
    if (success) {
      console.log('\n✅ Tests completed successfully!')
      process.exit(0)
    } else {
      console.log('\n❌ Tests failed!')
      process.exit(1)
    }
  })
  .catch((error) => {
    console.error('\n💥 Error running tests:', error.message)
    process.exit(1)
  })
