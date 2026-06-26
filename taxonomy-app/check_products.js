const { createClient } = require('@supabase/supabase-js');

const supabaseUrl = 'http://127.0.0.1:54331';
const supabaseKey = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZS1kZW1vIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImV4cCI6MTk4MzgxMjk5Nn0.EGIM96RAZx35lJzdJsyH-qQwv8Hdp7fsn3W0YpN81IU';
const supabase = createClient(supabaseUrl, supabaseKey);

async function checkProducts() {
  console.log("Checking 'products' table...");
  const { count: total, error: errTotal } = await supabase.from('products').select('*', { count: 'exact', head: true });
  console.log(`Total products: ${total}`);

  const { count: approved, error: errApproved } = await supabase.from('products').select('*', { count: 'exact', head: true }).eq('status', 'approved');
  console.log(`Approved products: ${approved}`);
  
  const { count: pending, error: errPending } = await supabase.from('products').select('*', { count: 'exact', head: true }).eq('status', 'pending');
  console.log(`Pending products: ${pending}`);
}

checkProducts();
