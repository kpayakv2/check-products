const { createClient } = require('@supabase/supabase-js');

// กำหนด Supabase URL และ Key
const supabaseUrl = 'http://127.0.0.1:54321';
const supabaseKey = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZS1kZW1vIiwicm9sZSI6ImFub24iLCJleHAiOjE5ODM4MTI5OTZ9.CRXP1A7WOeoJeXxjNni43kdQwgnWNReilDMblYTn_I0';

// สร้าง Supabase client
const supabase = createClient(supabaseUrl, supabaseKey);

async function testDatabaseConnection() {
    try {
        console.log('🔌 กำลังทดสอบการเชื่อมต่อ Supabase Local Database...');
        
        // ทดสอบ 1: ดูตารางที่มีอยู่
        console.log('\n� ทดสอบ 1: ดูตารางที่มีในฐานข้อมูล');
        const { data: tables, error: tableError } = await supabase
            .rpc('get_table_names');
            
        if (tableError) {
            console.log('⚠️  ไม่สามารถเรียกใช้ RPC ได้ - ลองตารางที่คาดว่ามี...');
            
            // ลองตารางที่น่าจะมี
            const possibleTables = [
                'product_categories', 'categories', 'product_taxonomy',
                'lemmas', 'terms', 'classification_rules', 'regex_rules'
            ];
            
            for (const tableName of possibleTables) {
                try {
                    const { data, error } = await supabase
                        .from(tableName)
                        .select('*', { count: 'exact', head: true });
                        
                    if (!error) {
                        console.log(`✅ ตาราง '${tableName}': ${data?.length || 0} รายการ`);
                    }
                } catch (err) {
                    console.log(`❌ ตาราง '${tableName}': ไม่พบ`);
                }
            }
        }

        // ทดสอบการเชื่อมต่อพื้นฐาน
        console.log('\n🔍 ทดสอบ 2: การเชื่อมต่อพื้นฐาน');
        const { data: healthCheck, error: healthError } = await supabase
            .from('information_schema.tables')
            .select('table_name')
            .eq('table_schema', 'public');
            
        if (healthError) {
            console.log('⚠️  ไม่สามารถดู information_schema ได้');
            console.log('Error:', healthError.message);
        } else {
            console.log(`✅ พบตารางใน public schema: ${healthCheck?.length || 0} ตาราง`);
            healthCheck?.forEach(table => {
                console.log(`   - ${table.table_name}`);
            });
        }

        console.log('\n🎉 การทดสอบเสร็จสมบูรณ์!');
        
    } catch (error) {
        console.error('\n❌ เกิดข้อผิดพลาด:', error.message);
        console.error('รายละเอียด:', error);
    }
}

// เริ่มทดสอบ
testDatabaseConnection();