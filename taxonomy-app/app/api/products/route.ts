import { NextRequest, NextResponse } from 'next/server'
import { DatabaseService } from '@/utils/supabase'

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url)
    const status = searchParams.get('status') as any
    const limit = parseInt(searchParams.get('limit') || '50')

    const products = await DatabaseService.getProducts(status, limit)
    return NextResponse.json({ success: true, data: products })
  } catch (error) {
    console.error('Error fetching products:', error)
    return NextResponse.json(
      { success: false, error: 'Failed to fetch products' },
      { status: 500 }
    )
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { 
      name_th, 
      name_en, 
      description, 
      category_id, 
      brand, 
      model, 
      sku, 
      price,
      attributes 
    } = body

    if (!name_th) {
      return NextResponse.json(
        { success: false, error: 'name_th is required' },
        { status: 400 }
      )
    }

    // Create product
    const product = await DatabaseService.createProduct({
      name_th,
      name_en,
      description,
      category_id,
      brand,
      model,
      sku,
      price,
      status: 'pending'
    })

    // Add attributes if provided
    if (attributes && attributes.length > 0) {
      for (const attr of attributes) {
        await DatabaseService.createProductAttribute({
          product_id: product.id,
          ...attr
        })
      }
    }

    return NextResponse.json({ success: true, data: product })
  } catch (error) {
    console.error('Error creating product:', error)
    return NextResponse.json(
      { success: false, error: 'Failed to create product' },
      { status: 500 }
    )
  }
}
