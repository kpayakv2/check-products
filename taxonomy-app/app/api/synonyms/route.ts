import { NextRequest, NextResponse } from 'next/server'
import { DatabaseService } from '@/utils/supabase'

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url)
    const categoryId = searchParams.get('category_id')

    const synonyms = await DatabaseService.getSynonyms(categoryId || undefined)
    return NextResponse.json({ success: true, data: synonyms })
  } catch (error) {
    console.error('Error fetching synonyms:', error)
    return NextResponse.json(
      { success: false, error: 'Failed to fetch synonyms' },
      { status: 500 }
    )
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { name, description, category_id, terms } = body

    if (!name || !terms || !Array.isArray(terms) || terms.length === 0) {
      return NextResponse.json(
        { success: false, error: 'name and terms array are required' },
        { status: 400 }
      )
    }

    const synonym = await DatabaseService.createSynonym({
      name,
      description,
      category_id,
      is_active: true
    })

    return NextResponse.json({ success: true, data: synonym })
  } catch (error) {
    console.error('Error creating synonym:', error)
    return NextResponse.json(
      { success: false, error: 'Failed to create synonym' },
      { status: 500 }
    )
  }
}
