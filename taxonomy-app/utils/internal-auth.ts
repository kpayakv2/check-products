/**
 * Internal shared-secret auth helpers.
 * Uses only Web Crypto (crypto.subtle) so this module works identically
 * in both the Edge middleware runtime and Node API route handlers.
 *
 * NOT a session system, despite the cookie name. There is no login in this
 * project (`auth.users` is empty and nothing calls `supabase.auth.*`), so the
 * token below is a pure function of the shared secret: the same value for every
 * person, every browser, every unlock, with no server-side record of what was
 * issued. A copied cookie therefore works forever, and the only revocation is
 * rotating INTERNAL_API_SECRET, which signs everyone out at once.
 * Treat the cookie value as the password itself. See CLAUDE.md → Authentication.
 */

export const SESSION_COOKIE_NAME = 'internal_session'

async function sha256Hex(input: string): Promise<string> {
  const data = new TextEncoder().encode(input)
  const digest = await crypto.subtle.digest('SHA-256', data)
  return Array.from(new Uint8Array(digest))
    .map((b) => b.toString(16).padStart(2, '0'))
    .join('')
}

export async function computeSessionToken(secret: string): Promise<string> {
  return sha256Hex(secret)
}

/**
 * Constant-time string comparison (no early return on first mismatch)
 * to avoid leaking secret length/prefix via response timing.
 */
export function constantTimeEqual(a: string, b: string): boolean {
  const len = Math.max(a.length, b.length)
  let diff = a.length === b.length ? 0 : 1

  for (let i = 0; i < len; i++) {
    const charA = i < a.length ? a.charCodeAt(i) : 0
    const charB = i < b.length ? b.charCodeAt(i) : 0
    diff |= charA ^ charB
  }

  return diff === 0
}

export async function isValidSessionCookie(cookieValue: string | undefined | null): Promise<boolean> {
  const secret = process.env.INTERNAL_API_SECRET
  if (!secret || !cookieValue) return false

  const expected = await computeSessionToken(secret)
  return constantTimeEqual(cookieValue, expected)
}
