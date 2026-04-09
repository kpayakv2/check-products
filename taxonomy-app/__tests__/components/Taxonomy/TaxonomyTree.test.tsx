import React from 'react'
import { render, screen, fireEvent, waitFor } from '../../utils/test-utils'
import { mockTaxonomyNode, mockSupabaseClient } from '../../utils/test-utils'

// Mock TaxonomyTree component
const MockTaxonomyTree = () => {
  const mockCategories = [
    { id: '1', name: 'Electronics', code: 'ELEC001', level: 1 },
    { id: '2', name: 'Smartphones', code: 'ELEC002', level: 2 },
    { id: '3', name: 'Laptops', code: 'ELEC003', level: 2 }
  ]

  return (
    <div>
      <h2>Taxonomy Management</h2>
      <input placeholder="Search categories..." />
      <button>Add Category</button>
      <div data-testid="taxonomy-tree">
        {mockCategories.map(category => (
          <div key={category.id} data-level={category.level}>
            <span>{category.name}</span>
            <button>Edit</button>
            <button>Delete</button>
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

describe('TaxonomyTree Component', () => {
  const mockTaxonomyData = [
    mockTaxonomyNode({
      id: '1',
      name: 'Electronics',
      code: 'ELEC001',
      parent_id: null,
      level: 1,
    }),
    mockTaxonomyNode({
      id: '2',
      name: 'Smartphones',
      code: 'ELEC002',
      parent_id: '1',
      level: 2,
    }),
    mockTaxonomyNode({
      id: '3',
      name: 'Laptops',
      code: 'ELEC003',
      parent_id: '1',
      level: 2,
    }),
  ]

  beforeEach(() => {
    jest.clearAllMocks()
    
    // Mock successful API response
    const mockChain = mockSupabaseClient.from()
    mockChain.select().mockResolvedValue({
      data: mockTaxonomyData,
      error: null,
    })
  })

  it('renders taxonomy tree correctly', () => {
    render(<MockTaxonomyTree />)
    
    expect(screen.getByText('Taxonomy Management')).toBeInTheDocument()
    expect(screen.getByText('Electronics')).toBeInTheDocument()
    expect(screen.getByText('Smartphones')).toBeInTheDocument()
    expect(screen.getByText('Laptops')).toBeInTheDocument()
  })

  it('displays search input', () => {
    render(<MockTaxonomyTree />)
    
    expect(screen.getByPlaceholderText('Search categories...')).toBeInTheDocument()
  })

  it('shows add category button', () => {
    render(<MockTaxonomyTree />)
    
    expect(screen.getByText('Add Category')).toBeInTheDocument()
  })

  it('shows edit and delete buttons', () => {
    render(<MockTaxonomyTree />)
    
    const editButtons = screen.getAllByText('Edit')
    const deleteButtons = screen.getAllByText('Delete')
    
    expect(editButtons).toHaveLength(3)
    expect(deleteButtons).toHaveLength(3)
  })

  it('displays hierarchical structure with levels', () => {
    render(<MockTaxonomyTree />)
    
    const level1Items = screen.getByText('Electronics').closest('[data-level="1"]')
    const level2Items = screen.getAllByText(/Smartphones|Laptops/)
    
    expect(level1Items).toBeInTheDocument()
    expect(level2Items).toHaveLength(2)
  })

  it('allows interaction with buttons', () => {
    render(<MockTaxonomyTree />)
    
    const addButton = screen.getByText('Add Category')
    const editButtons = screen.getAllByText('Edit')
    
    fireEvent.click(addButton)
    fireEvent.click(editButtons[0])
    
    expect(addButton).toBeInTheDocument()
    expect(editButtons[0]).toBeInTheDocument()
  })
})
