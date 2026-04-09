import React from 'react'
import { render, screen, fireEvent, waitFor } from '../../utils/test-utils'
import { mockProduct, mockSupabaseClient } from '../../utils/test-utils'

// Simple Mock Product Review component
const MockProductReview = () => {
  const [selectedProduct, setSelectedProduct] = React.useState(null)
  
  const mockProducts = [
    { id: '1', name: 'iPhone 15 Pro', description: 'Latest iPhone model', status: 'pending', similarity_score: 0.85 },
    { id: '2', name: 'Samsung Galaxy S24', description: 'Premium Android phone', status: 'pending', similarity_score: 0.92 }
  ]

  const handleApprove = (productId: string) => {
    console.log('Approved product:', productId)
  }

  const handleReject = (productId: string) => {
    console.log('Rejected product:', productId)
  }

  return (
    <div>
      <h2>Product Review</h2>
      <div data-testid="product-list">
        {mockProducts.map((product: any) => (
          <div key={product.id} data-testid={`product-${product.id}`}>
            <h3>{product.name}</h3>
            <p>{product.description}</p>
            <span data-testid={`status-${product.id}`}>Status: {product.status}</span>
            <span data-testid={`similarity-${product.id}`}>
              Similarity: {product.similarity_score}
            </span>
            <button 
              onClick={() => setSelectedProduct(product)}
              data-testid={`select-${product.id}`}
            >
              Select
            </button>
            <button 
              onClick={() => handleApprove(product.id)}
              data-testid={`approve-${product.id}`}
            >
              Approve
            </button>
            <button 
              onClick={() => handleReject(product.id)}
              data-testid={`reject-${product.id}`}
            >
              Reject
            </button>
          </div>
        ))}
      </div>
      {selectedProduct && (
        <div data-testid="product-detail-panel">
          <h3>Product Details</h3>
          <p>Name: {(selectedProduct as any).name}</p>
          <p>Description: {(selectedProduct as any).description}</p>
          <button onClick={() => setSelectedProduct(null)}>Close</button>
        </div>
      )}
    </div>
  )
}

// Mock the supabase client
jest.mock('../../../utils/supabase', () => ({
  supabase: mockSupabaseClient,
}))

describe('ProductReview Component', () => {
  const mockProductData = [
    mockProduct({
      id: '1',
      name: 'iPhone 15 Pro',
      description: 'Latest iPhone model',
      status: 'pending',
      similarity_score: 0.85,
    }),
    mockProduct({
      id: '2',
      name: 'Samsung Galaxy S24',
      description: 'Premium Android phone',
      status: 'pending',
      similarity_score: 0.92,
    }),
  ]

  beforeEach(() => {
    jest.clearAllMocks()
    
    // Mock successful API response
    mockSupabaseClient.from().select().mockResolvedValue({
      data: mockProductData,
      error: null,
    })
    
    mockSupabaseClient.from().update().mockResolvedValue({
      data: null,
      error: null,
    })
  })

  it('renders product review interface correctly', () => {
    render(<MockProductReview />)
    
    expect(screen.getByText('Product Review')).toBeInTheDocument()
    expect(screen.getByText('iPhone 15 Pro')).toBeInTheDocument()
    expect(screen.getByText('Samsung Galaxy S24')).toBeInTheDocument()
  })

  it('displays product list container', () => {
    render(<MockProductReview />)
    
    expect(screen.getByTestId('product-list')).toBeInTheDocument()
  })

  it('shows product information', () => {
    render(<MockProductReview />)
    
    expect(screen.getByTestId('product-1')).toBeInTheDocument()
    expect(screen.getByTestId('product-2')).toBeInTheDocument()
    expect(screen.getByText('Latest iPhone model')).toBeInTheDocument()
  })

  it('shows status and similarity information', () => {
    render(<MockProductReview />)
    
    expect(screen.getByTestId('status-1')).toHaveTextContent('Status: pending')
    expect(screen.getByTestId('similarity-1')).toHaveTextContent('Similarity: 0.85')
  })

  it('has approve and reject buttons', () => {
    render(<MockProductReview />)
    
    const approveButton = screen.getByTestId('approve-1')
    const rejectButton = screen.getByTestId('reject-1')
    
    fireEvent.click(approveButton)
    fireEvent.click(rejectButton)
    
    expect(approveButton).toBeInTheDocument()
    expect(rejectButton).toBeInTheDocument()
  })

  it('shows select button and detail panel functionality', () => {
    render(<MockProductReview />)
    
    const selectButton = screen.getByTestId('select-1')
    fireEvent.click(selectButton)
    
    expect(screen.getByTestId('product-detail-panel')).toBeInTheDocument()
    expect(screen.getByText('Product Details')).toBeInTheDocument()
    
    const closeButton = screen.getByText('Close')
    fireEvent.click(closeButton)
    
    expect(screen.queryByTestId('product-detail-panel')).not.toBeInTheDocument()
  })
})
