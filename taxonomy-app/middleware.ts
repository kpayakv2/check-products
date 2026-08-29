import { NextRequest, NextResponse } from 'next/server'
import { SESSION_COOKIE_NAME, isValidSessionCookie } from '@/utils/internal-auth'

const SAFE_METHODS = new Set(['GET', 'HEAD', 'OPTIONS'])

// Requests that must remain reachable without an unlocked session.
const UNGATED_PATHS = new Set(['/api/unlock'])

// Reads that go through the service role and therefore bypass RLS. A GET here
// hands out exactly the rows the anon key is denied, so it needs the session too.
const GATED_READ_PATHS = new Set(['/api/settings', '/api/import/history'])

export async function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl

  if (UNGATED_PATHS.has(pathname)) {
    return NextResponse.next()
  }

  if (SAFE_METHODS.has(request.method) && !GATED_READ_PATHS.has(pathname)) {
    return NextResponse.next()
  }

  const cookie = request.cookies.get(SESSION_COOKIE_NAME)?.value
  const valid = await isValidSessionCookie(cookie)

  if (!valid) {
    return NextResponse.json(
      { success: false, error: 'Unauthorized' },
      { status: 401 }
    )
  }

  return NextResponse.next()
}

export const config = {
  matcher: ['/api/:path*'],
}
