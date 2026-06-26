const fs = require('fs');
const { createClient } = require('@supabase/supabase-js');
require('dotenv').config({ path: '.env.local' });

const supabaseUrl = 'http://127.0.0.1:54331';
const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY || 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZS1kZW1vIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImV4cCI6MTk4MzgxMjk5Nn0.EGIM96RAZx35lJzdJsyH-qQwv8Hdp7fsn3W0YpN81IU';
const supabase = createClient(supabaseUrl, supabaseKey);

async function importGoldenDataset() {
  try {
    console.log('📖 Reading Golden Dataset...');
    const content = fs.readFileSync('../../input/รายการสินค้าพร้อมหมวดหมู่_AI.txt', 'utf16le');
    const lines = content.split('\n').filter(l => l.trim() !== '');
    
    const headers = lines[0].split('\t').map(h => h.trim());
    const skuIndex = headers.indexOf('itemid');
    const mainCatIndex = headers.indexOf('หมวดหมู่หลัก');
    const subCatIndex = headers.indexOf('หมวดย่อย');

    const records = [];
    for (let i = 1; i < lines.length; i++) {
      const cols = lines[i].split('\t').map(c => c.trim());
      if (cols.length > Math.max(skuIndex, mainCatIndex, subCatIndex)) {
        records.push({
          sku: cols[skuIndex],
          mainCat: cols[mainCatIndex],
          subCat: cols[subCatIndex]
        });
      }
    }
    
    console.log(`Found ${records.length} records in the dataset.`);

    // 1. Fetch existing taxonomy nodes
    console.log('Fetching taxonomy nodes from database...');
    const { data: nodes, error: nodesError } = await supabase.from('taxonomy_nodes').select('*');
    if (nodesError) throw nodesError;
    
    const nodeMap = new Map(); // id -> node
    const nameMap = new Map(); // name_th -> id (for main categories)
    const parentSubMap = new Map(); // parent_id_subCatName -> id
    
    for (const n of nodes) {
      nodeMap.set(n.id, n);
      if (n.level === 0) {
        nameMap.set(n.name_th, n.id);
      } else if (n.level === 1) {
        parentSubMap.set(`${n.parent_id}_${n.name_th}`, n.id);
      }
    }

    console.log(`Currently ${nodes.length} nodes in DB.`);

    // 2. Prepare updates for products
    console.log('Preparing to update products...');
    const productUpdates = [];
    let missingCats = 0;
    
    for (const r of records) {
      if (!r.sku || !r.mainCat) continue;
      
      let targetCatId = null;
      const mainId = nameMap.get(r.mainCat);
      
      if (!mainId) {
        console.warn(`Warning: Main category '${r.mainCat}' not found in DB.`);
        missingCats++;
        continue;
      }
      
      if (r.subCat) {
        targetCatId = parentSubMap.get(`${mainId}_${r.subCat}`);
        if (!targetCatId) {
          console.warn(`Warning: Sub category '${r.subCat}' under '${r.mainCat}' not found. Falling back to main category.`);
          targetCatId = mainId;
        }
      } else {
        targetCatId = mainId;
      }
      
      if (targetCatId) {
        productUpdates.push({
          sku: r.sku,
          category_id: targetCatId,
          status: 'approved' 
        });
      }
    }
    
    if (missingCats > 0) {
      console.log(`Warning: ${missingCats} records skipped due to missing categories.`);
    }

    // 3. Batch update products
    console.log(`Updating ${productUpdates.length} products...`);
    const BATCH_SIZE = 500;
    let updatedCount = 0;
    
    for (let i = 0; i < productUpdates.length; i += BATCH_SIZE) {
      const batch = productUpdates.slice(i, i + BATCH_SIZE);
      
      const chunkUpdates = batch.map(p => 
        supabase.from('products').update({ 
          category_id: p.category_id, 
          status: p.status 
        }).eq('sku', p.sku)
      );
      
      await Promise.all(chunkUpdates);
      
      updatedCount += batch.length;
      console.log(`Processed ${updatedCount}/${productUpdates.length} products`);
    }

    console.log('✅ Golden Dataset categories applied successfully!');

  } catch (err) {
    console.error('❌ Error during import:', err);
    process.exit(1);
  }
}

importGoldenDataset();
