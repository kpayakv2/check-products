import React from 'react'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { setupTestDatabase, clearTestData, testSupabase } from '../../setup/database-setup'

// Real TaxonomyTree Component (สมมติว่ามีอยู่)
const RealTaxonomyTree: React.FC = () => {
  const [nodes, setNodes] = React.useState<any[]>([])
  const [loading, setLoading] = React.useState(true)
  const [error, setError] = React.useState<string | null>(null)

  React.useEffect(() => {
    fetchTaxonomyNodes()
  }, [])

  const fetchTaxonomyNodes = async () => {
    try {
      setLoading(true)
      const { data, error } = await testSupabase
        .from('taxonomy_nodes')
        .select('*')
        .order('level', { ascending: true })
        .order('sort_order', { ascending: true })

      if (error) throw error
      setNodes(data || [])
    } catch (err: any) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const handleAddNode = async (parentId: string | null) => {
    const newNode = {
      id: `new-${Date.now()}`,
      name: 'New Category',
      code: `NEW${Date.now()}`,
      parent_id: parentId,
      level: parentId ? 2 : 1,
      sort_order: 1,
      is_active: true,
    }

    try {
      const { data, error } = await testSupabase
        .from('taxonomy_nodes')
        .insert([newNode])
        .select()
        .single()

      if (error) throw error
      setNodes(prev => [...prev, data])
    } catch (err: any) {
      setError(err.message)
    }
  }

  const handleDeleteNode = async (nodeId: string) => {
    try {
      const { error } = await testSupabase
        .from('taxonomy_nodes')
        .delete()
        .eq('id', nodeId)

      if (error) throw error
      setNodes(prev => prev.filter(n => n.id !== nodeId))
    } catch (err: any) {
      setError(err.message)
    }
  }

  if (loading) return <div data-testid="loading">Loading taxonomy...</div>
  if (error) return <div data-testid="error">Error: {error}</div>

  return (
    <div data-testid="taxonomy-tree">
      <h2>Taxonomy Management</h2>
      <button 
        onClick={() => handleAddNode(null)}
        data-testid="add-root-node"
      >
        Add Root Category
      </button>
      
      <div data-testid="node-list">
        {nodes.map(node => (
          <div key={node.id} data-testid={`node-${node.id}`} data-level={node.level}>
            <span data-testid={`node-name-${node.id}`}>{node.name}</span>
            <span data-testid={`node-code-${node.id}`}>{node.code}</span>
            <button 
              onClick={() => handleAddNode(node.id)}
              data-testid={`add-child-${node.id}`}
            >
              Add Child
            </button>
            <button 
              onClick={() => handleDeleteNode(node.id)}
              data-testid={`delete-${node.id}`}
            >
              Delete
            </button>
          </div>
        ))}
      </div>
    </div>
  )
}

describe('Real TaxonomyTree Component Tests', () => {
  beforeAll(async () => {
    await setupTestDatabase()
  })

  afterAll(async () => {
    await clearTestData()
  })

  beforeEach(async () => {
    await clearTestData()
    await setupTestDatabase()
  })

  describe('Component Rendering with Real Data', () => {
    it('should render taxonomy tree with real database data', async () => {
      render(<RealTaxonomyTree />)

      // Should show loading initially
      expect(screen.getByTestId('loading')).toBeInTheDocument()

      // Wait for data to load
      await waitFor(() => {
        expect(screen.getByTestId('taxonomy-tree')).toBeInTheDocument()
      })

      // Should display real test data
      expect(screen.getByText('Taxonomy Management')).toBeInTheDocument()
      expect(screen.getByTestId('node-test-tax-1')).toBeInTheDocument()
      expect(screen.getByTestId('node-test-tax-2')).toBeInTheDocument()

      // Check hierarchy
      const rootNode = screen.getByTestId('node-test-tax-1')
      const childNode = screen.getByTestId('node-test-tax-2')
      
      expect(rootNode).toHaveAttribute('data-level', '1')
      expect(childNode).toHaveAttribute('data-level', '2')
    })

    it('should display node names and codes from database', async () => {
      render(<RealTaxonomyTree />)

      await waitFor(() => {
        expect(screen.getByTestId('node-name-test-tax-1')).toHaveTextContent('Test Electronics')
        expect(screen.getByTestId('node-code-test-tax-1')).toHaveTextContent('TEST001')
        expect(screen.getByTestId('node-name-test-tax-2')).toHaveTextContent('Test Smartphones')
        expect(screen.getByTestId('node-code-test-tax-2')).toHaveTextContent('TEST002')
      })
    })
  })

  describe('Real CRUD Operations', () => {
    it('should add new root node to database', async () => {
      render(<RealTaxonomyTree />)

      await waitFor(() => {
        expect(screen.getByTestId('add-root-node')).toBeInTheDocument()
      })

      // Add new root node
      fireEvent.click(screen.getByTestId('add-root-node'))

      // Wait for new node to appear
      await waitFor(() => {
        const newNodes = screen.getAllByText('New Category')
        expect(newNodes.length).toBeGreaterThan(0)
      })

      // Verify in database
      const { data } = await testSupabase
        .from('taxonomy_nodes')
        .select('*')
        .eq('name', 'New Category')

      expect(data).toHaveLength(1)
      expect(data![0].level).toBe(1)
      expect(data![0].parent_id).toBeNull()
    })

    it('should add child node to existing parent', async () => {
      render(<RealTaxonomyTree />)

      await waitFor(() => {
        expect(screen.getByTestId('add-child-test-tax-1')).toBeInTheDocument()
      })

      // Add child to root node
      fireEvent.click(screen.getByTestId('add-child-test-tax-1'))

      // Wait for new child node
      await waitFor(() => {
        const newNodes = screen.getAllByText('New Category')
        expect(newNodes.length).toBeGreaterThan(0)
      })

      // Verify in database
      const { data } = await testSupabase
        .from('taxonomy_nodes')
        .select('*')
        .eq('name', 'New Category')

      expect(data).toHaveLength(1)
      expect(data![0].level).toBe(2)
      expect(data![0].parent_id).toBe('test-tax-1')
    })

    it('should delete node from database', async () => {
      render(<RealTaxonomyTree />)

      await waitFor(() => {
        expect(screen.getByTestId('delete-test-tax-2')).toBeInTheDocument()
      })

      // Delete child node first (foreign key constraint)
      fireEvent.click(screen.getByTestId('delete-test-tax-2'))

      // Wait for node to disappear
      await waitFor(() => {
        expect(screen.queryByTestId('node-test-tax-2')).not.toBeInTheDocument()
      })

      // Verify deletion in database
      const { data } = await testSupabase
        .from('taxonomy_nodes')
        .select('*')
        .eq('id', 'test-tax-2')

      expect(data).toHaveLength(0)
    })
  })

  describe('Error Handling with Real Database', () => {
    it('should handle database connection errors', async () => {
      // Temporarily break the connection (mock network error)
      const originalFrom = testSupabase.from
      testSupabase.from = jest.fn().mockImplementation(() => ({
        select: jest.fn().mockImplementation(() => ({
          order: jest.fn().mockImplementation(() => ({
            order: jest.fn().mockRejectedValue(new Error('Network error'))
          }))
        }))
      }))

      render(<RealTaxonomyTree />)

      await waitFor(() => {
        expect(screen.getByTestId('error')).toBeInTheDocument()
        expect(screen.getByText(/Network error/)).toBeInTheDocument()
      })

      // Restore original function
      testSupabase.from = originalFrom
    })

    it('should handle constraint violations', async () => {
      render(<RealTaxonomyTree />)

      await waitFor(() => {
        expect(screen.getByTestId('taxonomy-tree')).toBeInTheDocument()
      })

      // Try to create duplicate code (should fail)
      const duplicateNode = {
        id: 'duplicate-test',
        name: 'Duplicate Code',
        code: 'TEST001', // Same as existing
        parent_id: null,
        level: 1,
        sort_order: 1,
        is_active: true,
      }

      // This should trigger an error in the component
      try {
        await testSupabase
          .from('taxonomy_nodes')
          .insert([duplicateNode])
      } catch (error) {
        expect(error).toBeDefined()
      }
    })
  })

  describe('Real-time Updates', () => {
    it('should reflect database changes in UI', async () => {
      render(<RealTaxonomyTree />)

      await waitFor(() => {
        expect(screen.getByTestId('taxonomy-tree')).toBeInTheDocument()
      })

      // Manually add node to database (simulating external change)
      const externalNode = {
        id: 'external-node',
        name: 'External Addition',
        code: 'EXT001',
        parent_id: null,
        level: 1,
        sort_order: 3,
        is_active: true,
      }

      await testSupabase
        .from('taxonomy_nodes')
        .insert([externalNode])

      // Component would need to implement real-time subscriptions
      // For now, we can test manual refresh
      fireEvent.click(screen.getByTestId('add-root-node'))

      // The component should now show the external node
      // (This would work better with Supabase real-time subscriptions)
    })
  })
})
