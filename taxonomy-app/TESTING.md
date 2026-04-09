# Testing Guide

## Overview

This project uses a comprehensive testing strategy with multiple layers of testing to ensure code quality and reliability.

## Testing Stack

- **Unit Testing**: Jest + React Testing Library
- **API Testing**: Jest + Supertest
- **E2E Testing**: Playwright
- **Coverage**: Jest Coverage Reports
- **CI/CD**: GitHub Actions

## Test Structure

```
__tests__/
├── utils/
│   ├── test-utils.tsx          # React testing utilities
│   └── api-test-utils.ts       # API testing utilities
├── components/
│   ├── Taxonomy/
│   │   └── TaxonomyTree.test.tsx
│   ├── Synonym/
│   │   └── SynonymManager.test.tsx
│   └── Product/
│       └── ProductReview.test.tsx
└── api/
    ├── taxonomy/
    │   └── route.test.ts
    ├── synonyms/
    │   └── route.test.ts
    └── products/
        └── route.test.ts

e2e/
├── taxonomy-management.spec.ts
├── synonym-management.spec.ts
└── product-review.spec.ts
```

## Running Tests

### Unit Tests

```bash
# Run all unit tests
npm run test

# Run tests in watch mode
npm run test:watch

# Run tests with coverage
npm run test:coverage

# Run tests for CI (no watch, with coverage)
npm run test:ci
```

### E2E Tests

```bash
# Install Playwright browsers (first time only)
npx playwright install

# Run E2E tests
npm run test:e2e

# Run E2E tests with UI
npm run test:e2e:ui

# Run specific test file
npx playwright test taxonomy-management.spec.ts

# Run tests in headed mode (see browser)
npx playwright test --headed

# Debug tests
npx playwright test --debug
```

## Test Coverage

Our coverage targets:
- **Branches**: 80%
- **Functions**: 80%
- **Lines**: 80%
- **Statements**: 80%

### Viewing Coverage Reports

```bash
# Generate coverage report
npm run test:coverage

# Open coverage report in browser
open coverage/lcov-report/index.html
```

## Writing Tests

### Unit Test Guidelines

1. **Test Structure**: Use Arrange-Act-Assert pattern
2. **Mocking**: Mock external dependencies (Supabase, APIs)
3. **Isolation**: Each test should be independent
4. **Descriptive Names**: Use clear, descriptive test names

#### Example Unit Test

```typescript
import { render, screen, fireEvent, waitFor } from '../../utils/test-utils'
import TaxonomyTree from '../../../components/Taxonomy/TaxonomyTree'
import { mockTaxonomyNode } from '../../utils/test-utils'

describe('TaxonomyTree Component', () => {
  beforeEach(() => {
    jest.clearAllMocks()
  })

  it('renders taxonomy tree correctly', async () => {
    // Arrange
    const mockData = [mockTaxonomyNode()]
    mockSupabaseClient.from().select().mockResolvedValue({
      data: mockData,
      error: null,
    })

    // Act
    render(<TaxonomyTree />)

    // Assert
    await waitFor(() => {
      expect(screen.getByText('Taxonomy Management')).toBeInTheDocument()
    })
  })
})
```

### API Test Guidelines

1. **Mock Supabase**: Use mock client for database operations
2. **Test All Methods**: GET, POST, PUT, DELETE
3. **Validation**: Test input validation and error handling
4. **Status Codes**: Verify correct HTTP status codes

#### Example API Test

```typescript
import { createMockReqRes, mockSupabaseForAPI } from '../../utils/api-test-utils'

describe('/api/taxonomy', () => {
  it('creates new taxonomy node successfully', async () => {
    // Arrange
    const newNodeData = {
      name: 'New Category',
      code: 'NEW001',
    }
    const { req, res } = createMockReqRes({
      method: 'POST',
      body: newNodeData,
    })

    // Act
    await taxonomyHandler(req, res)

    // Assert
    expect(res._getStatusCode()).toBe(201)
    const response = JSON.parse(res._getData())
    expect(response.success).toBe(true)
  })
})
```

### E2E Test Guidelines

1. **User Flows**: Test complete user workflows
2. **Real Interactions**: Use actual browser interactions
3. **Wait Strategies**: Use proper waiting strategies
4. **Page Objects**: Consider using page object pattern for complex flows

#### Example E2E Test

```typescript
import { test, expect } from '@playwright/test'

test.describe('Taxonomy Management', () => {
  test('should create new taxonomy category', async ({ page }) => {
    // Navigate to page
    await page.goto('/taxonomy')
    
    // Interact with UI
    await page.click('[data-testid="add-category-btn"]')
    await page.fill('[data-testid="category-name-input"]', 'Test Category')
    await page.click('[data-testid="save-category-btn"]')
    
    // Verify result
    await expect(page.locator('text=Test Category')).toBeVisible()
  })
})
```

## Test Data Management

### Mock Data

Use the provided mock data generators:

```typescript
import { 
  mockTaxonomyNode,
  mockSynonym,
  mockProduct 
} from '../utils/test-utils'

const testNode = mockTaxonomyNode({
  name: 'Custom Name',
  code: 'CUSTOM001'
})
```

### Test Database

For integration tests, consider using:
- Supabase test database
- Docker containers for isolated testing
- In-memory database for unit tests

## Continuous Integration

### GitHub Actions

Our CI pipeline runs:
1. **Linting**: ESLint checks
2. **Type Checking**: TypeScript compilation
3. **Unit Tests**: Jest with coverage
4. **E2E Tests**: Playwright tests
5. **Security Scan**: Dependency audit
6. **Build**: Next.js build verification

### Coverage Reports

Coverage reports are:
- Generated on every CI run
- Uploaded to Codecov
- Displayed in pull requests
- Required to meet minimum thresholds

## Debugging Tests

### Unit Tests

```bash
# Debug specific test
npm test -- --testNamePattern="should create new category"

# Debug with Node debugger
node --inspect-brk node_modules/.bin/jest --runInBand

# Run single test file
npm test TaxonomyTree.test.tsx
```

### E2E Tests

```bash
# Debug mode (step through tests)
npx playwright test --debug

# Headed mode (see browser)
npx playwright test --headed

# Trace viewer (after test run)
npx playwright show-trace trace.zip
```

## Best Practices

### General

1. **Test Pyramid**: More unit tests, fewer E2E tests
2. **Fast Feedback**: Unit tests should run quickly
3. **Reliable**: Tests should not be flaky
4. **Maintainable**: Keep tests simple and focused

### React Testing

1. **Test Behavior**: Test what users see and do
2. **Avoid Implementation Details**: Don't test internal state
3. **Accessibility**: Use accessible queries (getByRole, getByLabelText)
4. **User Events**: Use userEvent for realistic interactions

### API Testing

1. **Test Contracts**: Verify request/response formats
2. **Error Handling**: Test error conditions
3. **Edge Cases**: Test boundary conditions
4. **Security**: Test authentication and authorization

### E2E Testing

1. **Critical Paths**: Focus on important user journeys
2. **Data Independence**: Tests should not depend on specific data
3. **Cleanup**: Clean up test data after tests
4. **Parallel Execution**: Design tests to run in parallel

## Troubleshooting

### Common Issues

1. **Timeout Errors**: Increase timeout or improve wait strategies
2. **Flaky Tests**: Add proper waits and stabilize test conditions
3. **Mock Issues**: Ensure mocks are properly reset between tests
4. **Coverage Gaps**: Add tests for uncovered code paths

### Getting Help

1. Check test logs and error messages
2. Review test documentation
3. Ask team members for guidance
4. Create issues for persistent problems

## Test Maintenance

### Regular Tasks

1. **Update Dependencies**: Keep testing libraries updated
2. **Review Coverage**: Monitor coverage trends
3. **Refactor Tests**: Keep tests clean and maintainable
4. **Performance**: Monitor test execution time

### Code Reviews

When reviewing test code:
1. Verify test logic and assertions
2. Check for proper mocking
3. Ensure good test coverage
4. Review test naming and structure
