import { NextRequest, NextResponse } from 'next/server'
import { DatabaseService } from '@/utils/supabase'

export async function PUT(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    const body = await request.json()
    const { id } = params

    const synonym = await DatabaseService.updateSynonym(id, body)
    return NextResponse.json({ success: true, data: synonym })
  } catch (error) {
    console.error('Error updating synonym:', error)
    return NextResponse.json(
      { success: false, error: 'Failed to update synonym' },
      { status: 500 }
    )
  }
}

export async function DELETE(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    const { id } = params
    await DatabaseService.deleteSynonym(id)
    return NextResponse.json({ success: true })
  } catch (error) {
    console.error('Error deleting synonym:', error)
    return NextResponse.json(
      { success: false, error: 'Failed to delete synonym' },
      { status: 500 }
    )
  }
}
