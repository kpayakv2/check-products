const { createClient } = require('@supabase/supabase-js')
require('dotenv').config({ path: 'taxonomy-app/.env.local' })

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY // ต้องใช้ Service Role เพื่อลบข้อมูล

if (!supabaseUrl || !supabaseServiceKey) {
  console.error('Missing Supabase credentials in .env.local')
  process.exit(1)
}

const supabase = createClient(supabaseUrl, supabaseServiceKey)

async function cleanupUnanalyzedProducts() {
  console.log('🚀 เริ่มการลบสินค้าที่ยังไม่ผ่านการวิเคราะห์ (1,620 รายการ)...')
  
  // ลบข้อมูล
  const { data, error, count } = await supabase
    .from('products')
    .delete({ count: 'exact' })
    .or('confidence_score.eq.0,embedding.is.null')

  if (error) {
    console.error('❌ เกิดข้อผิดพลาดในการลบ:', error.message)
  } else {
    console.log(`✅ ลบสำเร็จทั้งหมด: ${count} รายการ`)
  }
}

cleanupUnanalyzedProducts()
