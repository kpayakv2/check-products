import { NextRequest, NextResponse } from 'next/server'
import { DatabaseService } from '@/utils/supabase'

export async function GET(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    const { id } = params
    const matches = await DatabaseService.getSimilarityMatches(id)
    return NextResponse.json({ success: true, data: matches })
  } catch (error) {
    console.error('Error fetching similarity matches:', error)
    return NextResponse.json(
      { success: false, error: 'Failed to fetch similarity matches' },
      { status: 500 }
    )
  }
}

export async function POST(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    const body = await request.json()
    const { id } = params
    const { target_product_id, similarity_score, match_type } = body

    if (!target_product_id || !similarity_score) {
      return NextResponse.json(
        { success: false, error: 'target_product_id and similarity_score are required' },
        { status: 400 }
      )
    }

    const match = await DatabaseService.createSimilarityMatch({
      product_a_id: id,
      product_b_id: target_product_id,
      similarity_score,
      match_type: match_type || 'semantic'
    })

    return NextResponse.json({ success: true, data: match })
  } catch (error) {
    console.error('Error creating similarity match:', error)
    return NextResponse.json(
      { success: false, error: 'Failed to create similarity match' },
      { status: 500 }
    )
  }
}
