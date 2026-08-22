import { NextRequest, NextResponse } from 'next/server'
import { z } from 'zod'
import { rateLimit, getClientIP } from '@/utils/rate-limit'
import { withErrorHandling } from '@/utils/error-handler'
import { validateRequest } from '@/utils/validation'
import { SESSION_COOKIE_NAME, computeSessionToken, constantTimeEqual } from '@/utils/internal-auth'

const UnlockSchema = z.object({
  secret: z.string().min(1, 'กรุณากรอกรหัสผ่าน')
})

// Stricter than the general write-route limiter — this endpoint is the
// brute-force target for the shared secret.
const limiter = rateLimit({
  interval: 60 * 1000,
  uniqueTokenPerInterval: 500,
})

/**
 * POST /api/unlock
 * ตรวจสอบ shared secret แล้วออก httpOnly session cookie
 * สำหรับปลดล็อกการเข้าถึง API routes ที่แก้ไขข้อมูลได้
 */
export async function POST(request: NextRequest): Promise<NextResponse> {
  return withErrorHandling(async () => {
    try {
      await limiter.check(5, `unlock-${getClientIP(request)}`)
    } catch {
      return NextResponse.json(
        { success: false, error: 'Too many requests' },
        { status: 429 }
      )
    }

    const body = await validateRequest(request, UnlockSchema)
    if ('error' in body) {
      return NextResponse.json(body, { status: 400 })
    }

    const expectedSecret = process.env.INTERNAL_API_SECRET
    if (!expectedSecret) {
      console.error('INTERNAL_API_SECRET is not configured')
      return NextResponse.json(
        { success: false, error: 'Server misconfigured' },
        { status: 500 }
      )
    }

    if (!constantTimeEqual(body.secret, expectedSecret)) {
      return NextResponse.json(
        { success: false, error: 'รหัสผ่านไม่ถูกต้อง' },
        { status: 401 }
      )
    }

    const token = await computeSessionToken(expectedSecret)
    const response = NextResponse.json({ success: true })

    // No `secure` flag: this tool is served over plain HTTP on the office LAN.
    response.cookies.set(SESSION_COOKIE_NAME, token, {
      httpOnly: true,
      sameSite: 'lax',
      path: '/',
      maxAge: 60 * 60 * 24 * 30, // 30 days
    })

    return response
  }, 'POST /api/unlock')
}
