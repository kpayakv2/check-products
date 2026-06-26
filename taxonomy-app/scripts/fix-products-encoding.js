const { createClient } = require('@supabase/supabase-js');
const fs = require('fs');
const path = require('path');

// Initialize Supabase client using local anon key and URL
const supabaseUrl = 'http://127.0.0.1:54331';
const supabaseKey = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZS1kZW1vIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImV4cCI6MTk4MzgxMjk5Nn0.EGIM96RAZx35lJzdJsyH-qQwv8Hdp7fsn3W0YpN81IU';
const supabase = createClient(supabaseUrl, supabaseKey);

async function repairProducts() {
  try {
    const filePath = path.resolve(__dirname, '../../recovered_products.json');
    console.log(`Reading from: ${filePath}`);
    const data = JSON.parse(fs.readFileSync(filePath, 'utf8'));
    console.log(`Loaded ${data.length} products from JSON.`);

    // Check how many have ? characters currently
    const { data: dbData, error: countError } = await supabase
      .from('products')
      .select('sku, name_th');
      
    if (countError) throw countError;
    
    let corruptedCount = 0;
    for (const p of dbData) {
      if (p.name_th && p.name_th.includes('?')) {
        corruptedCount++;
      }
    }
    console.log(`Found ${corruptedCount} corrupted products with '?' in DB.`);

    // Prepare updates
    console.log('Preparing batch update...');
    
    // Batch updates to avoid request payload limits
    const BATCH_SIZE = 500;
    let updatedCount = 0;
    
    for (let i = 0; i < data.length; i += BATCH_SIZE) {
      const batch = data.slice(i, i + BATCH_SIZE).map(item => ({
        sku: item.sku,
        name_th: item.name_th,
        price: item.price,
        // we can also pass status, etc, but name_th is the main issue
        // wait, we don't want to overwrite category_id if it was manually set
        // but if we just want to repair name_th, we can query by SKU.
      }));
      
      // Since supabase upsert requires all primary keys or uses unique constraints,
      // 'sku' is the unique column in products table (probably). 
      // If 'id' is the primary key and sku is unique, we can use onConflict: 'sku'
      const { error } = await supabase
        .from('products')
        .upsert(batch, { onConflict: 'sku', ignoreDuplicates: false });
        
      if (error) {
        console.error(`Error in batch ${i}:`, error.message);
        throw error;
      }
      
      updatedCount += batch.length;
      console.log(`Updated ${updatedCount}/${data.length} products...`);
    }

    console.log('✅ Repair completed successfully.');

  } catch (err) {
    console.error('❌ Error during repair:', err);
    process.exit(1);
  }
}

repairProducts();
