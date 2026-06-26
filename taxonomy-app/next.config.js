/** @type {import('next').NextConfig} */
const nextConfig = {
  images: {
    domains: ['images.unsplash.com'],
  },
  // สำหรับ font optimization
  optimizeFonts: true,
  output: 'standalone',

  // Proxy FastAPI + Supabase — ทำให้เครื่อง LAN อื่นเรียก backend/db ผ่าน Next.js ได้
  // browser เรียก /api/fastapi/... → Next.js server ส่งต่อ → 127.0.0.1:8000/...
  // browser เรียก /api/supabase/... → Next.js server ส่งต่อ → 127.0.0.1:54331/...
  async rewrites() {
    const fastapiUrl = process.env.FASTAPI_URL || 'http://127.0.0.1:8000'
    const supabaseUrl = process.env.SUPABASE_URL || 'http://127.0.0.1:54331'
    return [
      {
        source: '/api/fastapi/:path*',
        destination: `${fastapiUrl}/:path*`,
      },
      {
        source: '/api/supabase/:path*',
        destination: `${supabaseUrl}/:path*`,
      },
    ]
  },
}

module.exports = nextConfig
