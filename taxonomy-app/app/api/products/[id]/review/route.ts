import { NextRequest, NextResponse } from 'next/server'
import { DatabaseService } from '@/utils/supabase'

export async function POST(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    const body = await request.json()
    const { id } = params
    const { status, reviewer_id, comments } = body

    if (!status || !['approved', 'rejected'].includes(status)) {
      return NextResponse.json(
        { success: false, error: 'Invalid status. Must be approved or rejected' },
        { status: 400 }
      )
    }

    // Update product status
    const product = await DatabaseService.updateProductStatus(id, status, reviewer_id)

    // Create review history
    await DatabaseService.createReviewHistory({
      product_id: id,
      reviewer_id,
      action: status,
      comments
    })

    return NextResponse.json({ success: true, data: product })
  } catch (error) {
    console.error('Error reviewing product:', error)
    return NextResponse.json(
      { success: false, error: 'Failed to review product' },
      { status: 500 }
    )
  }
}
