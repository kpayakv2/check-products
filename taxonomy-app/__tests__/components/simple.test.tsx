import React from 'react'
import { render, screen } from '@testing-library/react'
import '@testing-library/jest-dom'

// Simple test component
const TestComponent = ({ title }: { title: string }) => {
  return <h1>{title}</h1>
}

describe('Simple Component Test', () => {
  it('renders component correctly', () => {
    render(<TestComponent title="Test Title" />)
    
    const heading = screen.getByText('Test Title')
    expect(heading).toBeInTheDocument()
  })

  it('renders with different props', () => {
    render(<TestComponent title="Another Title" />)
    
    const heading = screen.getByText('Another Title')
    expect(heading).toBeInTheDocument()
  })
})

// Test utilities
describe('Test Utilities', () => {
  it('should have basic test setup working', () => {
    expect(true).toBe(true)
  })

  it('should handle async operations', async () => {
    const promise = Promise.resolve('test')
    const result = await promise
    expect(result).toBe('test')
  })
})
