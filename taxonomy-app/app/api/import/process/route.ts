import { NextRequest, NextResponse } from 'next/server'
import { DatabaseService } from '@/utils/supabase'
import { supabaseAdmin } from '@/utils/supabase-admin'

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData()
    const filePath = formData.get('filePath') as string
    const importId = formData.get('importId') as string
    const mappingStr = formData.get('columnMapping') as string
    const columnMapping = mappingStr ? JSON.parse(mappingStr) : null

    if (!filePath || !importId) return NextResponse.json({ error: 'Missing params' }, { status: 400 })

    const encoder = new TextEncoder()
    const stream = new ReadableStream({
      async start(controller) {
        const send = (data: any) => controller.enqueue(encoder.encode(JSON.stringify(data) + '\n'))

        try {
          // 1. Download & Clean
          send({ type: 'progress', processed: 0, total: 100, last_item: 'กำลังดึงไฟล์...' })
          
          const { data: fileData, error: dlErr } = await supabaseAdmin.storage.from('uploads').download(filePath)
          if (dlErr || !fileData) throw new Error('Download failed')

          const buffer = await fileData.arrayBuffer()
          const text = new TextDecoder('utf-8').decode(buffer)
          
          // CRUCIAL: Remove BOM and strange chars before split
          const cleanText = text.replace(/^[\uFEFF\u200B\u00A0]/, '').trim()
          const lines = cleanText.split(/\r\n|\r|\n/).filter(line => line.trim().length > 0)
          
          let sep = lines[0].includes(';') ? ';' : (lines[0].includes('\t') ? '\t' : ',')
          const parse = (l: string) => {
            const res: string[] = []; let c = ''; let q = false
            for (let i = 0; i < l.length; i++) {
              if (l[i] === '"') q = !q
              else if (l[i] === sep && !q) { res.push(c.trim()); c = '' }
              else c += l[i]
            }
            res.push(c.trim()); return res
          }

          const headers = parse(lines[0]).map(h => h.replace(/[\uFEFF]/g, '').trim())
          
          // FIND PRODUCT NAME COLUMN (More aggressive search)
          let pIdx = -1
          if (typeof columnMapping?.product_name_index === 'number') pIdx = columnMapping.product_name_index
          else {
            pIdx = headers.indexOf(columnMapping?.product_name || 'รายการ')
            if (pIdx === -1) pIdx = headers.findIndex(h => h.includes('รายการ') || h.includes('สินค้า') || h.toLowerCase().includes('name'))
          }
          if (pIdx === -1) pIdx = headers.length > 1 ? 1 : 0

          const productNames = lines.slice(1).map(l => parse(l)[pIdx]).filter(n => n && n.length > 1)
          
          if (productNames.length === 0) throw new Error(`ไม่พบคอลัมน์ชื่อสินค้า (Index: ${pIdx})`)
          
          // Update DB Status
          await supabaseAdmin.from('imports').update({ total_records: productNames.length, status: 'processing' }).eq('id', importId)

          // 2. Sequential AI Processing with Immediate Feedback
          for (let i = 0; i < productNames.length; i++) {
            const name = productNames[i]
            
            // 🔥 SEND FEEDBACK IMMEDIATELY BEFORE HEAVY WORK
            send({ type: 'progress', processed: i + 1, total: productNames.length, last_item: name })

            try {
              // Create clean name for AI
              const cleanName = name.replace(/[^\u0E00-\u0E7Fa-zA-Z0-9\s]/g, ' ').replace(/\s+/g, ' ').trim()
              
              // Call AI Functions (Sequential for stability)
              const embRes = await supabaseAdmin.functions.invoke('generate-embeddings-local', { body: { text: cleanName } })
              const aiRes = await supabaseAdmin.functions.invoke('hybrid-classification-local', { body: { product_name: name } })

              const emb = embRes.data?.embedding
              const sugg = aiRes.data?.top_suggestion
              const conf = sugg?.confidence || 0
              const status = conf >= 0.8 ? 'approved' : 'pending_review_category'

              // Save to DB
              await supabaseAdmin.from('products').insert({
                name_th: name,
                category_id: sugg?.category_id || null,
                status: status,
                confidence_score: conf,
                embedding: emb,
                import_batch_id: importId,
                metadata: { clean_name: cleanName, suggested_name: sugg?.category_name }
              })

            } catch (err) {
              console.error(`Row ${i} failed:`, err.message)
            }

            // Small delay to allow UI to render and server to breathe
            if (i % 5 === 0) await new Promise(r => setTimeout(r, 20))
          }

          // 3. Complete
          await supabaseAdmin.from('imports').update({ status: 'completed', processed_records: productNames.length, completed_at: new Date().toISOString() }).eq('id', importId)
          send({ type: 'completed', total: productNames.length })
          controller.close()

        } catch (err) {
          console.error('API Stream Error:', err.message)
          send({ type: 'error', message: err.message })
          controller.close()
        }
      }
    })

    return new Response(stream, { headers: { 'Content-Type': 'text/plain; charset=utf-8', 'Transfer-Encoding': 'chunked' } })
  } catch (err) {
    return NextResponse.json({ error: err.message }, { status: 500 })
  }
}
