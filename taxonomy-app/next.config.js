/** @type {import('next').NextConfig} */
const nextConfig = {
  experimental: {
    appDir: true,
  },
  images: {
    domains: ['images.unsplash.com'],
  },
  // สำหรับ font optimization
  optimizeFonts: true,
  output: 'standalone',
}

module.exports = nextConfig
