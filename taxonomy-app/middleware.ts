import { NextRequest, NextResponse } from 'next/server'
import { SESSION_COOKIE_NAME, isValidSessionCookie } from '@/utils/internal-auth'

/**
 * This is the *only* real access gate in the project. The 42 RLS policies that
 * reference auth.role()/auth.uid() can never be satisfied — nobody logs in — and
 * every write goes through the service role, which bypasses RLS outright.
 * So a route that slips past this file is unprotected. See CLAUDE.md → Authentication.
 */
const SAFE_METHODS = new Set(['GET', 'HEAD', 'OPTIONS'])

// API requests that must remain reachable without an unlocked session.
// /api/lock ต้องอยู่ในนี้ด้วย — ไม่งั้นคุกกี้ที่หมดอายุไปแล้วจะกันปุ่ม "ออกจากระบบ" เองไม่ให้ทำงาน
const UNGATED_API_PATHS = new Set(['/api/unlock', '/api/lock'])

// Reads that go through the service role and therefore bypass RLS. A GET here
// hands out exactly the rows the anon key is denied, so it needs the session too.
const GATED_READ_PATHS = new Set(['/api/settings', '/api/import/history'])

// หน้าเว็บที่ต้องเข้าได้เสมอแม้ยังไม่ปลดล็อก — มีแค่หน้ากรอกรหัสเอง
// ไม่งั้นคนที่ยังไม่ปลดล็อกจะโดนเด้งกลับมา /unlock วนลูปไม่รู้จบ
const UNGATED_PAGES = new Set(['/unlock'])

export async function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl

  if (pathname.startsWith('/api/')) {
    if (UNGATED_API_PATHS.has(pathname)) {
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

  // หน้าเว็บ (ไม่ใช่ /api/*) — เดิมเข้าได้เสมอเพราะ matcher ครอบแค่ /api/:path*
  // ตอนนี้ต้องปลดล็อกก่อนเหมือนกัน ไม่งั้นเปิดเว็บครั้งแรกแล้วเห็นข้อมูลได้เลยโดยไม่กรอกรหัส
  if (UNGATED_PAGES.has(pathname)) {
    return NextResponse.next()
  }

  const cookie = request.cookies.get(SESSION_COOKIE_NAME)?.value
  const valid = await isValidSessionCookie(cookie)

  if (!valid) {
    const unlockUrl = new URL('/unlock', request.url)
    unlockUrl.searchParams.set('next', pathname)
    return NextResponse.redirect(unlockUrl)
  }

  return NextResponse.next()
}

export const config = {
  matcher: ['/((?!_next/static|_next/image|favicon.ico).*)'],
}
