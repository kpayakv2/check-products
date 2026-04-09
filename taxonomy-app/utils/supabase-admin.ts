import { createClient } from '@supabase/supabase-js'

// ⚠️ กุญแจนี้มีสิทธิ์สูงสุด ห้ามใช้ในส่วนที่เป็น Client-side (Browser)
// ใช้ได้เฉพาะใน API Routes หรือ Server Components เท่านั้น
const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY!

if (!supabaseServiceKey) {
  console.warn('⚠️ SUPABASE_SERVICE_ROLE_KEY is missing! API calls might fail.')
}

export const supabaseAdmin = createClient(supabaseUrl, supabaseServiceKey || '', {
  auth: {
    autoRefreshToken: false,
    persistSession: false
  }
})
