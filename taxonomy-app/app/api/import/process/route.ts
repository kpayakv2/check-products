import { NextRequest, NextResponse } from 'next/server'
import { DatabaseService } from '@/utils/supabase'
import { supabaseAdmin } from '@/utils/supabase-admin'
import Papa from 'papaparse'

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
          // 1. Fetch Current Status (for Resume)
          const { data: currentImport } = await supabaseAdmin.from('imports').select('*').eq('id', importId).single()
          const startFrom = currentImport?.metadata?.last_processed_index || 0
          let errorLog = currentImport?.metadata?.error_log || []
          let successCount = currentImport?.success_records || 0
          let pendingCount = currentImport?.metadata?.pending_records || 0

          send({ type: 'progress', processed: startFrom, total: 100, last_item: 'กำลังดึงไฟล์...' })

          const { data: fileData, error: dlErr } = await supabaseAdmin.storage.from('uploads').download(filePath)
          if (dlErr || !fileData) throw new Error('Download failed')

          const buffer = await fileData.arrayBuffer()
          const text = new TextDecoder('utf-8').decode(buffer)

          // 1.2 Parse CSV
          const parseResult = Papa.parse(text, {
            header: true,
            skipEmptyLines: true,
            transformHeader: (h) => h.replace(/^[\uFEFF\u200B\u00A0]/, '').trim()
          })

          if (parseResult.errors.length > 0 && parseResult.data.length === 0) {
            throw new Error(`CSV Parse Error: ${parseResult.errors[0].message}`)
          }

          const headers = parseResult.meta.fields || []
          const rows = parseResult.data as any[]

          // FIND PRODUCT NAME COLUMN
          let productNameKey = ''
          if (typeof columnMapping?.product_name === 'string' && columnMapping.product_name !== '') {
             productNameKey = columnMapping.product_name
          } else {
             productNameKey = headers.find(h => 
               h === 'รายการ' || 
               h === 'สินค้า' || 
               h.toLowerCase().includes('product') || 
               h.toLowerCase().includes('name')
             ) || headers[0]
          }

          console.log(`[Import ${importId}] Using column "${productNameKey}" as product name`)

          // Update DB Status
          await supabaseAdmin.from('imports').update({ 
            total_records: rows.length, 
            status: 'processing',
            metadata: {
              ...(currentImport?.metadata || {}),
              product_name_column: productNameKey
            }
          }).eq('id', importId)

          // 2. Parallel Processing with Resume Capability
          const BATCH_SIZE = 5
          let processedCount = startFrom

          for (let i = startFrom; i < rows.length; i += BATCH_SIZE) {
            const batchRows = rows.slice(i, i + BATCH_SIZE)
            const batchErrors: any[] = []
            
            await Promise.all(batchRows.map(async (row, index) => {
              const name = String(row[productNameKey] || '').trim()
              if (!name) return

              const currentIndex = i + index + 1
              if (index === 0) send({ type: 'progress', processed: currentIndex, total: rows.length, last_item: name })

              try {
                const cleanName = name.replace(/[^\u0E00-\u0E7Fa-zA-Z0-9\s]/g, ' ').replace(/\s+/g, ' ').trim()

                // Call AI
                const embRes = await supabaseAdmin.functions.invoke('generate-embeddings-local', { body: { text: cleanName } })
                const aiRes = await supabaseAdmin.functions.invoke('hybrid-classification-local', { body: { product_name: name } })

                const emb = embRes.data?.embedding
                const sugg = aiRes.data?.top_suggestion
                const conf = sugg?.confidence || 0
                
                // DEDUPLICATION
                let potentialDuplicates = []
                if (emb) {
                   const { data: duplicates } = await supabaseAdmin.rpc('match_products_by_embedding', {
                     query_embedding: emb,
                     match_threshold: 0.15,
                     match_count: 3
                   })
                   potentialDuplicates = duplicates || []
                }

                const isPossibleDuplicate = potentialDuplicates.length > 0
                const status = (conf >= 0.8 && !isPossibleDuplicate) ? 'approved' : 'pending_review_category'

                if (status === 'approved') {
                  const { error: insErr } = await supabaseAdmin.from('products').insert({
                    name_th: name, category_id: sugg?.category_id || null,
                    status: 'approved', confidence_score: conf, embedding: emb, import_batch_id: importId,
                    metadata: { clean_name: cleanName, suggested_name: sugg?.category_name, source_row: row, is_auto_approved: true }
                  })
                  if (!insErr) successCount++
                } else {
                  const { error: sugErr } = await supabaseAdmin.from('product_category_suggestions').insert({
                    suggested_category_id: sugg?.category_id || null, confidence_score: conf, suggestion_method: 'hybrid_ai_preview',
                    metadata: {
                      product_name: name, cleaned_name: cleanName, tokens: aiRes.data?.analysis?.tokens || [],
                      units: aiRes.data?.analysis?.units || [], attributes: aiRes.data?.analysis?.attributes || {},
                      explanation: aiRes.data?.analysis?.explanation || '', import_id: importId, source_row: row,
                      potential_duplicates: potentialDuplicates, is_duplicate_detected: isPossibleDuplicate
                    }
                  })
                  if (!sugErr) pendingCount++
                }
              } catch (err) {
                console.error(`Row ${currentIndex} failed:`, err)
                batchErrors.push({ index: currentIndex, name, error: String(err) })
              }
            }))

            processedCount += batchRows.length
            errorLog = [...errorLog, ...batchErrors]

            // 💾 PERSIST PROGRESS TO DATABASE (Check-pointing)
            await supabaseAdmin.from('imports').update({ 
              processed_records: processedCount,
              success_records: successCount,
              error_records: errorLog.length,
              metadata: { 
                ...(currentImport?.metadata || {}),
                product_name_column: productNameKey,
                last_processed_index: processedCount,
                error_log: errorLog,
                pending_records: pendingCount
              }
            }).eq('id', importId)

            send({ type: 'progress', processed: Math.min(processedCount, rows.length), total: rows.length, last_item: `Batch completed: ${processedCount}/${rows.length}` })
          }

          // 3. Complete & Analytics
          const { data: finalProducts } = await supabaseAdmin.from('products').select('confidence_score').eq('import_batch_id', importId)
          const avgConfidence = finalProducts && finalProducts.length > 0 
            ? finalProducts.reduce((acc, p) => acc + (p.confidence_score || 0), 0) / finalProducts.length 
            : 0

          await supabaseAdmin.from('imports').update({ 
            status: 'completed', 
            processed_records: rows.length, 
            success_records: successCount,
            error_records: errorLog.length,
            completed_at: new Date().toISOString(),
            metadata: {
              ...(currentImport?.metadata || {}),
              product_name_column: productNameKey,
              last_processed_index: rows.length,
              error_log: errorLog,
              avg_confidence: avgConfidence,
              total_errors: errorLog.length,
              pending_records: pendingCount
            }
          }).eq('id', importId)
          
          send({ type: 'completed', total: rows.length, avg_confidence: avgConfidence, errors: errorLog.length, success: successCount, pending: pendingCount })
          controller.close()

        } catch (err) {
          const errorMessage = err instanceof Error ? err.message : String(err)
          send({ type: 'error', message: errorMessage })
          controller.close()
        }
      }
    })

    return new Response(stream, { headers: { 'Content-Type': 'text/plain; charset=utf-8', 'Transfer-Encoding': 'chunked' } })
  } catch (err) {
    const errorMessage = err instanceof Error ? err.message : String(err)
    return NextResponse.json({ error: errorMessage }, { status: 500 })
  }
}
