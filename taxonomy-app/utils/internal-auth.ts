/**
 * Internal shared-secret auth helpers.
 * Uses only Web Crypto (crypto.subtle) so this module works identically
 * in both the Edge middleware runtime and Node API route handlers.
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
