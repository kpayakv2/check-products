import React, { ReactElement } from 'react'
import { render, RenderOptions } from '@testing-library/react'
import { Toaster } from 'react-hot-toast'

// Mock providers for testing
const AllTheProviders = ({ children }: { children: React.ReactNode }) => {
  return (
    <>
      {children}
      <Toaster />
    </>
  )
}

const customRender = (
  ui: ReactElement,
  options?: Omit<RenderOptions, 'wrapper'>,
) => render(ui, { wrapper: AllTheProviders, ...options })

export * from '@testing-library/react'
export { customRender as render }

// Mock data generators
export const mockTaxonomyNode = (overrides = {}) => ({
  id: 'test-node-1',
  name: 'Test Category',
  code: 'TEST001',
  parent_id: null,
  level: 1,
  sort_order: 1,
  is_active: true,
  created_at: new Date().toISOString(),
  updated_at: new Date().toISOString(),
  ...overrides,
})

export const mockSynonym = (overrides = {}) => ({
  id: 'test-synonym-1',
  lemma: 'test lemma',
  category_id: 'test-category-1',
  confidence_score: 0.95,
  is_verified: true,
  created_at: new Date().toISOString(),
  updated_at: new Date().toISOString(),
  ...overrides,
})

export const mockSynonymTerm = (overrides = {}) => ({
  id: 'test-term-1',
  synonym_id: 'test-synonym-1',
  term: 'test term',
  language: 'th',
  created_at: new Date().toISOString(),
  ...overrides,
})

export const mockProduct = (overrides = {}) => ({
  id: 'test-product-1',
  name: 'Test Product',
  description: 'Test product description',
  category_id: 'test-category-1',
  status: 'pending',
  similarity_score: 0.85,
  created_at: new Date().toISOString(),
  updated_at: new Date().toISOString(),
  ...overrides,
})

export const mockApiResponse = (data: any, error: any = null) => ({
  data,
  error,
  count: Array.isArray(data) ? data.length : 1,
  status: error ? 400 : 200,
  statusText: error ? 'Bad Request' : 'OK',
})

// Test helpers
export const waitForLoadingToFinish = () => 
  new Promise(resolve => setTimeout(resolve, 0))

// Mock Supabase client with proper Jest mocking
export const mockSupabaseClient = {
  from: jest.fn().mockReturnValue({
    select: jest.fn().mockReturnValue({
      order: jest.fn().mockReturnValue({
        mockResolvedValue: jest.fn().mockResolvedValue({ data: [], error: null })
      }),
      mockResolvedValue: jest.fn().mockResolvedValue({ data: [], error: null })
    }),
    insert: jest.fn().mockReturnValue({
      select: jest.fn().mockReturnValue({
        single: jest.fn().mockResolvedValue({ data: null, error: null })
      }),
      mockResolvedValue: jest.fn().mockResolvedValue({ data: [], error: null })
    }),
    update: jest.fn().mockReturnValue({
      eq: jest.fn().mockReturnValue({
        select: jest.fn().mockReturnValue({
          single: jest.fn().mockResolvedValue({ data: null, error: null })
        }),
        mockResolvedValue: jest.fn().mockResolvedValue({ data: null, error: null })
      }),
      mockResolvedValue: jest.fn().mockResolvedValue({ data: null, error: null })
    }),
    delete: jest.fn().mockReturnValue({
      eq: jest.fn().mockReturnValue({
        mockResolvedValue: jest.fn().mockResolvedValue({ data: null, error: null })
      }),
      mockResolvedValue: jest.fn().mockResolvedValue({ data: null, error: null })
    })
  }),
  auth: {
    getUser: jest.fn().mockResolvedValue({ data: { user: null }, error: null }),
    signIn: jest.fn().mockResolvedValue({ data: null, error: null }),
    signOut: jest.fn().mockResolvedValue({ error: null }),
  },
  storage: {
    from: jest.fn().mockReturnValue({
      upload: jest.fn().mockResolvedValue({ data: null, error: null }),
      download: jest.fn().mockResolvedValue({ data: null, error: null }),
    })
  },
  rpc: jest.fn().mockResolvedValue({ data: null, error: null }),
}

// Add a dummy test to prevent Jest error
describe('Test Utils', () => {
  it('should export test utilities', () => {
    expect(mockTaxonomyNode).toBeDefined()
    expect(mockSynonym).toBeDefined()
    expect(mockProduct).toBeDefined()
    expect(mockSupabaseClient).toBeDefined()
  })
})
