'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { toast } from 'react-hot-toast'
import { supabase } from '@/utils/supabase'
import { 
  FolderIcon, 
  FileTextIcon,
  CheckCircleIcon,
  ClockIcon,
  DownloadIcon
} from 'lucide-react'

interface StorageFile {
  name: string
  id: string
  updated_at: string
  created_at: string
  last_accessed_at: string
  metadata: any
}

interface StorageImportProps {
  onFileSelect?: (file: File, fileName: string) => void
}

export default function StorageImport({ onFileSelect }: StorageImportProps) {
  const [files, setFiles] = useState<StorageFile[]>([])
  const [loading, setLoading] = useState(false)
  const [downloading, setDownloading] = useState(false)
  const [selectedFile, setSelectedFile] = useState<string | null>(null)

  useEffect(() => {
    loadFiles()
  }, [])

  const loadFiles = async () => {
    setLoading(true)
    try {
      const { data, error } = await supabase.storage
        .from('uploads')
        .list('products', {
          limit: 20,
          offset: 0,
          sortBy: { column: 'created_at', order: 'desc' }
        })

      if (error) throw error

      setFiles(data || [])
    } catch (error) {
      console.error('Error loading files:', error)
      toast.error('ไม่สามารถโหลดไฟล์ได้')
    } finally {
      setLoading(false)
    }
  }

  const selectFile = async (fileName: string) => {
    if (!onFileSelect) return
    
    setDownloading(true)
    
    try {
      // Download file from Supabase Storage
      const { data, error } = await supabase.storage
        .from('uploads')
        .download(`products/${fileName}`)

      if (error) throw error
      if (!data) throw new Error('No file data received')

      // Convert Blob to File
      const file = new File([data], fileName, { type: data.type })
      
      toast.success(`เลือกไฟล์: ${fileName}`)
      onFileSelect(file, fileName)

    } catch (error) {
      console.error('Download error:', error)
      toast.error('ไม่สามารถดาวน์โหลดไฟล์ได้')
    } finally {
      setDownloading(false)
    }
  }

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString('th-TH')
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="premium-card p-6">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold text-gray-900 font-noto-sans-thai mb-2">
              📁 เลือกไฟล์จาก Storage
            </h2>
            <p className="text-gray-600">
              เลือกไฟล์ CSV ที่อัปโหลดจาก Product Similarity Checker
            </p>
          </div>
          <button
            onClick={loadFiles}
            disabled={loading}
            className="px-4 py-2 text-gray-600 hover:text-gray-800 transition-colors flex items-center space-x-2"
          >
            {loading ? (
              <ClockIcon className="w-4 h-4 animate-spin" />
            ) : (
              <DownloadIcon className="w-4 h-4" />
            )}
            <span>รีเฟรช</span>
          </button>
        </div>
      </div>

      {/* Files List */}
      <div className="premium-card">
        <h3 className="text-lg font-semibold mb-4 font-noto-sans-thai">
          📄 ไฟล์ที่พร้อมใช้งาน ({files.length} ไฟล์)
        </h3>
        
        {loading ? (
          <div className="text-center py-8">
            <ClockIcon className="w-8 h-8 animate-spin mx-auto mb-2 text-blue-500" />
            <p className="text-gray-600">กำลังโหลดไฟล์...</p>
          </div>
        ) : files.length === 0 ? (
          <div className="text-center py-8">
            <FolderIcon className="w-12 h-12 mx-auto mb-2 text-gray-400" />
            <p className="text-gray-600">ไม่พบไฟล์ในโฟลเดอร์ products</p>
          </div>
        ) : (
          <div className="space-y-3">
            {files.map((file) => (
              <motion.div
                key={file.name}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className={`p-4 border rounded-lg cursor-pointer transition-all ${
                  selectedFile === file.name
                    ? 'border-blue-500 bg-blue-50'
                    : 'border-gray-200 hover:border-gray-300'
                }`}
                onClick={() => setSelectedFile(file.name)}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    <FileTextIcon className="w-5 h-5 text-blue-500" />
                    <div>
                      <p className="font-medium text-gray-900">{file.name}</p>
                      <p className="text-sm text-gray-500">
                        อัปโหลดเมื่อ: {formatDate(file.created_at)}
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center space-x-3">
                    {selectedFile === file.name && (
                      <button
                        onClick={(e) => {
                          e.stopPropagation()
                          selectFile(file.name)
                        }}
                        disabled={downloading}
                        className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50 transition-colors text-sm flex items-center space-x-2"
                      >
                        {downloading ? (
                          <>
                            <ClockIcon className="w-4 h-4 animate-spin" />
                            <span>กำลังโหลด...</span>
                          </>
                        ) : (
                          <>
                            <CheckCircleIcon className="w-4 h-4" />
                            <span>เลือกไฟล์นี้</span>
                          </>
                        )}
                      </button>
                    )}
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
