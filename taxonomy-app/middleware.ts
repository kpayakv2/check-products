import { NextRequest, NextResponse } from 'next/server'
import { SESSION_COOKIE_NAME, isValidSessionCookie } from '@/utils/internal-auth'

/**
 * This is the *only* real access gate in the project. The 42 RLS policies that
 * reference auth.role()/auth.uid() can never be satisfied — nobody logs in — and
 * every write goes through the service role, which bypasses RLS outright.
 * So a route that slips past this file is unprotected. See CLAUDE.md → Authentication.
 */
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
