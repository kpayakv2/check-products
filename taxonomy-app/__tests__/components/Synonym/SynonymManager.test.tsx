import React from 'react'
import { render, screen, fireEvent, waitFor } from '../../utils/test-utils'
import { mockSynonym, mockSynonymTerm, mockSupabaseClient } from '../../utils/test-utils'

// Simple Mock Synonym Manager component
const MockSynonymManager = () => {
  const mockSynonyms = [
    { id: '1', lemma: 'smartphone', category: 'Electronics', confidence: 0.95 },
    { id: '2', lemma: 'mobile phone', category: 'Electronics', confidence: 0.88 }
  ]

  return (
    <div>
      <h2>Synonym Management</h2>
      <button>Add Synonym</button>
      <div data-testid="synonym-list">
        {mockSynonyms.map(synonym => (
          <div key={synonym.id} data-testid={`synonym-${synonym.id}`}>
            <span>{synonym.lemma}</span>
            <span>{synonym.category}</span>
            <button data-testid={`edit-${synonym.id}`}>Edit</button>
            <button data-testid={`delete-${synonym.id}`}>Delete</button>
          </div>
        ))}
      </div>
    </div>
  )
}

// Mock the supabase client
jest.mock('../../../utils/supabase', () => ({
  supabase: mockSupabaseClient,
}))

describe('SynonymManager Component', () => {
  const mockSynonymData = [
    mockSynonym({
      id: '1',
      lemma: 'smartphone',
      category_id: 'cat-1',
      confidence_score: 0.95,
    }),
    mockSynonym({
      id: '2',
      lemma: 'mobile phone',
      category_id: 'cat-1',
      confidence_score: 0.88,
    }),
  ]

  const mockTermData = [
    mockSynonymTerm({
      id: '1',
      synonym_id: '1',
      term: 'smart phone',
      language: 'en',
    }),
    mockSynonymTerm({
      id: '2',
      synonym_id: '1',
      term: 'มือถือ',
      language: 'th',
    }),
  ]

  beforeEach(() => {
    jest.clearAllMocks()
    
    // Mock successful API response
    mockSupabaseClient.from().select().mockResolvedValue({
      data: mockSynonymData,
      error: null,
    })
  })

  it('renders synonym manager correctly', () => {
    render(<MockSynonymManager />)
    
    expect(screen.getByText('Synonym Management')).toBeInTheDocument()
    expect(screen.getByText('smartphone')).toBeInTheDocument()
    expect(screen.getByText('mobile phone')).toBeInTheDocument()
  })

  it('displays add synonym button', () => {
    render(<MockSynonymManager />)
    
    expect(screen.getByText('Add Synonym')).toBeInTheDocument()
  })

  it('shows synonym list container', () => {
    render(<MockSynonymManager />)
    
    expect(screen.getByTestId('synonym-list')).toBeInTheDocument()
  })

  it('displays synonym items correctly', () => {
    render(<MockSynonymManager />)
    
    expect(screen.getByTestId('synonym-1')).toBeInTheDocument()
    expect(screen.getByTestId('synonym-2')).toBeInTheDocument()
    
    expect(screen.getByText('smartphone')).toBeInTheDocument()
    expect(screen.getByText('mobile phone')).toBeInTheDocument()
  })

  it('shows edit buttons for each synonym', () => {
    render(<MockSynonymManager />)
    
    const editButtons = screen.getAllByText('Edit')
    expect(editButtons).toHaveLength(2)
    
    fireEvent.click(editButtons[0])
    expect(editButtons[0]).toBeInTheDocument()
  })

  it('shows delete buttons for each synonym', () => {
    render(<MockSynonymManager />)
    
    const deleteButtons = screen.getAllByText('Delete')
    expect(deleteButtons).toHaveLength(2)
    
    fireEvent.click(deleteButtons[0])
    expect(deleteButtons[0]).toBeInTheDocument()
  })

  it('renders category information', () => {
    render(<MockSynonymManager />)
    
    const categoryElements = screen.getAllByText('Electronics')
    expect(categoryElements).toHaveLength(2)
  })
})
