'use client'

import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { DragDropContext, Droppable, Draggable, DropResult } from 'react-beautiful-dnd'
import { toast } from 'react-hot-toast'
import { 
  ChevronRightIcon, 
  ChevronDownIcon, 
  FolderIcon, 
  FolderOpenIcon,
  PlusIcon,
  EditIcon,
  TrashIcon,
  SearchIcon,
  CodeIcon,
  GripVerticalIcon,
  DownloadIcon,
  UploadIcon,
  CopyIcon,
  MoreVerticalIcon,
  FileTextIcon
} from 'lucide-react'
import { TaxonomyNode } from '@/utils/supabase'

interface EnhancedTaxonomyTreeProps {
  categories: TaxonomyNode[]
  onCategorySelect?: (category: TaxonomyNode) => void
  onCategoryAdd?: (parentId?: string) => void
  onCategoryEdit?: (category: TaxonomyNode) => void
  onCategoryDelete?: (category: TaxonomyNode) => void
  onCategoryMove?: (categoryId: string, newParentId?: string, newIndex?: number) => void
  onBulkImport?: (yamlData: string) => void
  onBulkExport?: () => void
  editable?: boolean
  searchable?: boolean
  className?: string
}

interface TreeNodeProps {
  category: TaxonomyNode
  level: number
  index: number
  isExpanded: boolean
  isSelected: boolean
  searchTerm: string
  onToggle: (id: string) => void
  onSelect: (category: TaxonomyNode) => void
  onEdit: (category: TaxonomyNode) => void
  onDelete: (category: TaxonomyNode) => void
  onAddChild: (parentId: string) => void
  onGenerateCode: (category: TaxonomyNode) => void
  editable: boolean
}

// Generate category code automatically
const generateCategoryCode = (name: string, level: number): string => {
  const cleanName = name.replace(/[^\u0E00-\u0E7Fa-zA-Z0-9]/g, '')
  const prefix = level === 0 ? 'CAT' : level === 1 ? 'SUB' : 'ITM'
  const hash = cleanName.split('').reduce((a, b) => {
    a = ((a << 5) - a) + b.charCodeAt(0)
    return a & a
  }, 0)
  return `${prefix}_${Math.abs(hash).toString(36).toUpperCase().substring(0, 6)}`
}

// Export to YAML format
const exportToYAML = (categories: TaxonomyNode[]): string => {
  const convertToYAML = (nodes: TaxonomyNode[], indent = 0): string => {
    return nodes.map(node => {
      const spaces = '  '.repeat(indent)
      let yaml = `${spaces}- id: "${node.id}"\n`
      yaml += `${spaces}  name_th: "${node.name_th}"\n`
      if (node.name_en) yaml += `${spaces}  name_en: "${node.name_en}"\n`
      if (node.description) yaml += `${spaces}  description: "${node.description}"\n`
      if (node.keywords && node.keywords.length > 0) {
        yaml += `${spaces}  keywords:\n`
        node.keywords.forEach(keyword => {
          yaml += `${spaces}    - "${keyword}"\n`
        })
      }
      yaml += `${spaces}  level: ${node.level}\n`
      yaml += `${spaces}  sort_order: ${node.sort_order}\n`
      if (node.children && node.children.length > 0) {
        yaml += `${spaces}  children:\n`
        yaml += convertToYAML(node.children, indent + 2)
      }
      return yaml
    }).join('')
  }

  return `# Taxonomy Export - ${new Date().toISOString()}\ntaxonomy:\n${convertToYAML(categories, 1)}`
}

const TreeNode: React.FC<TreeNodeProps> = ({
  category,
  level,
  index,
  isExpanded,
  isSelected,
  searchTerm,
  onToggle,
  onSelect,
  onEdit,
  onDelete,
  onAddChild,
  onGenerateCode,
  editable
}) => {
  const [showActions, setShowActions] = useState(false)
  const hasChildren = category.children && category.children.length > 0
  const isMatched = searchTerm && (
    category.name_th.toLowerCase().includes(searchTerm.toLowerCase()) ||
    category.name_en?.toLowerCase().includes(searchTerm.toLowerCase()) ||
    category.keywords?.some(k => k.toLowerCase().includes(searchTerm.toLowerCase()))
  )

  const highlightText = (text: string) => {
    if (!searchTerm) return text
    const regex = new RegExp(`(${searchTerm})`, 'gi')
    const parts = text.split(regex)
    return parts.map((part, i) => 
      regex.test(part) ? 
        <mark key={i} className="bg-yellow-200 px-1 rounded">{part}</mark> : 
        part
    )
  }

  return (
    <Draggable draggableId={category.id} index={index} isDragDisabled={!editable}>
      {(provided, snapshot) => (
        <div
          ref={provided.innerRef}
          {...provided.draggableProps}
          className={`
            group relative
            ${snapshot.isDragging ? 'opacity-50' : ''}
            ${isMatched ? 'ring-2 ring-yellow-300 rounded-lg' : ''}
          `}
        >
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className={`
              flex items-center py-2 px-3 rounded-lg cursor-pointer transition-all duration-200
              ${isSelected 
                ? 'bg-primary-50 border border-primary-200 text-primary-700' 
                : 'hover:bg-gray-50'
              }
              ${level > 0 ? `ml-${level * 6}` : ''}
            `}
            onClick={() => onSelect(category)}
            onMouseEnter={() => setShowActions(true)}
            onMouseLeave={() => setShowActions(false)}
          >
            {/* Drag Handle */}
            {editable && (
              <div {...provided.dragHandleProps} className="mr-2 opacity-0 group-hover:opacity-100">
                <GripVerticalIcon className="w-4 h-4 text-gray-400" />
              </div>
            )}

            {/* Expand/Collapse Button */}
            <button
              onClick={(e) => {
                e.stopPropagation()
                onToggle(category.id)
              }}
              className="mr-2 p-1 rounded hover:bg-gray-200 transition-colors"
            >
              {hasChildren ? (
                isExpanded ? (
                  <ChevronDownIcon className="w-4 h-4 text-gray-600" />
                ) : (
                  <ChevronRightIcon className="w-4 h-4 text-gray-600" />
                )
              ) : (
                <div className="w-4 h-4" />
              )}
            </button>

            {/* Folder Icon */}
            <div className="mr-3">
              {hasChildren ? (
                isExpanded ? (
                  <FolderOpenIcon className="w-5 h-5 text-blue-500" />
                ) : (
                  <FolderIcon className="w-5 h-5 text-blue-500" />
                )
              ) : (
                <FileTextIcon className="w-5 h-5 text-gray-400" />
              )}
            </div>

            {/* Category Info */}
            <div className="flex-1 min-w-0">
              <div className="flex items-center space-x-2">
                <span className="font-medium text-gray-900 truncate">
                  {highlightText(category.name_th)}
                </span>
                {category.name_en && (
                  <span className="text-sm text-gray-500 truncate">
                    ({highlightText(category.name_en)})
                  </span>
                )}
              </div>
              
              {category.keywords && category.keywords.length > 0 && (
                <div className="flex flex-wrap gap-1 mt-1">
                  {category.keywords.slice(0, 3).map((keyword, i) => (
                    <span key={i} className="text-xs bg-gray-100 text-gray-600 px-2 py-0.5 rounded">
                      {highlightText(keyword)}
                    </span>
                  ))}
                  {category.keywords.length > 3 && (
                    <span className="text-xs text-gray-400">+{category.keywords.length - 3}</span>
                  )}
                </div>
              )}
            </div>

            {/* Level Badge */}
            <span className="text-xs bg-gray-200 text-gray-600 px-2 py-1 rounded-full">
              L{category.level}
            </span>

            {/* Actions */}
            <AnimatePresence>
              {(showActions || isSelected) && editable && (
                <motion.div
                  initial={{ opacity: 0, scale: 0.8 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.8 }}
                  className="flex items-center space-x-1 ml-2"
                >
                  <button
                    onClick={(e) => {
                      e.stopPropagation()
                      onAddChild(category.id)
                    }}
                    className="p-1 rounded hover:bg-blue-100 text-blue-600"
                    title="เพิ่มหมวดหมู่ย่อย"
                  >
                    <PlusIcon className="w-4 h-4" />
                  </button>
                  
                  <button
                    onClick={(e) => {
                      e.stopPropagation()
                      onGenerateCode(category)
                    }}
                    className="p-1 rounded hover:bg-green-100 text-green-600"
                    title="สร้างรหัสอัตโนมัติ"
                  >
                    <CodeIcon className="w-4 h-4" />
                  </button>
                  
                  <button
                    onClick={(e) => {
                      e.stopPropagation()
                      onEdit(category)
                    }}
                    className="p-1 rounded hover:bg-yellow-100 text-yellow-600"
                    title="แก้ไข"
                  >
                    <EditIcon className="w-4 h-4" />
                  </button>
                  
                  <button
                    onClick={(e) => {
                      e.stopPropagation()
                      onDelete(category)
                    }}
                    className="p-1 rounded hover:bg-red-100 text-red-600"
                    title="ลบ"
                  >
                    <TrashIcon className="w-4 h-4" />
                  </button>
                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>

          {/* Children */}
          <AnimatePresence>
            {isExpanded && hasChildren && (
              <Droppable droppableId={`children-${category.id}`} type="CATEGORY">
                {(provided) => (
                  <motion.div
                    ref={provided.innerRef}
                    {...provided.droppableProps}
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    exit={{ opacity: 0, height: 0 }}
                    className="ml-4 border-l-2 border-gray-100 pl-4"
                  >
                    {category.children!.map((child, childIndex) => (
                      <TreeNode
                        key={child.id}
                        category={child}
                        level={level + 1}
                        index={childIndex}
                        isExpanded={isExpanded}
                        isSelected={isSelected}
                        searchTerm={searchTerm}
                        onToggle={onToggle}
                        onSelect={onSelect}
                        onEdit={onEdit}
                        onDelete={onDelete}
                        onAddChild={onAddChild}
                        onGenerateCode={onGenerateCode}
                        editable={editable}
                      />
                    ))}
                    {provided.placeholder}
                  </motion.div>
                )}
              </Droppable>
            )}
          </AnimatePresence>
        </div>
      )}
    </Draggable>
  )
}

export default function EnhancedTaxonomyTree({
  categories,
  onCategorySelect,
  onCategoryAdd,
  onCategoryEdit,
  onCategoryDelete,
  onCategoryMove,
  onBulkImport,
  onBulkExport,
  editable = true,
  searchable = true,
  className = ''
}: EnhancedTaxonomyTreeProps) {
  const [expandedNodes, setExpandedNodes] = useState<Set<string>>(new Set())
  const [selectedCategory, setSelectedCategory] = useState<TaxonomyNode | null>(null)
  const [searchTerm, setSearchTerm] = useState('')
  const [showImportModal, setShowImportModal] = useState(false)
  const [importYaml, setImportYaml] = useState('')

  const handleDragEnd = (result: DropResult) => {
    if (!result.destination || !onCategoryMove) return

    const { draggableId, destination, source } = result
    
    // If dropped in the same position, do nothing
    if (destination.droppableId === source.droppableId && destination.index === source.index) {
      return
    }

    const newParentId = destination.droppableId.replace('children-', '')
    onCategoryMove(draggableId, newParentId === 'root' ? undefined : newParentId, destination.index)
  }

  const handleToggle = (id: string) => {
    setExpandedNodes(prev => {
      const newSet = new Set(prev)
      if (newSet.has(id)) {
        newSet.delete(id)
      } else {
        newSet.add(id)
      }
      return newSet
    })
  }

  const handleSelect = (category: TaxonomyNode) => {
    setSelectedCategory(category)
    onCategorySelect?.(category)
  }

  const handleGenerateCode = (category: TaxonomyNode) => {
    const code = generateCategoryCode(category.name_th, category.level)
    navigator.clipboard.writeText(code)
    toast.success(`คัดลอกรหัส: ${code}`)
  }

  const handleExport = () => {
    const yaml = exportToYAML(categories)
    const blob = new Blob([yaml], { type: 'text/yaml' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `taxonomy-export-${new Date().toISOString().split('T')[0]}.yaml`
    a.click()
    URL.revokeObjectURL(url)
    toast.success('ส่งออก YAML สำเร็จ')
  }

  const handleImport = () => {
    if (!importYaml.trim()) {
      toast.error('กรุณาใส่ข้อมูล YAML')
      return
    }
    
    try {
      onBulkImport?.(importYaml)
      setShowImportModal(false)
      setImportYaml('')
      toast.success('นำเข้าข้อมูลสำเร็จ')
    } catch (error) {
      toast.error('เกิดข้อผิดพลาดในการนำเข้าข้อมูล')
    }
  }

  const filteredCategories = searchTerm 
    ? categories.filter(cat => 
        cat.name_th.toLowerCase().includes(searchTerm.toLowerCase()) ||
        cat.name_en?.toLowerCase().includes(searchTerm.toLowerCase()) ||
        cat.keywords?.some(k => k.toLowerCase().includes(searchTerm.toLowerCase()))
      )
    : categories

  return (
    <div className={`bg-white rounded-lg border border-gray-200 ${className}`}>
      {/* Header */}
      <div className="p-4 border-b border-gray-200">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-900">โครงสร้าง Taxonomy</h3>
          
          <div className="flex items-center space-x-2">
            {editable && (
              <>
                <button
                  onClick={() => setShowImportModal(true)}
                  className="btn-secondary text-sm"
                >
                  <UploadIcon className="w-4 h-4 mr-1" />
                  นำเข้า YAML
                </button>
                
                <button
                  onClick={handleExport}
                  className="btn-secondary text-sm"
                >
                  <DownloadIcon className="w-4 h-4 mr-1" />
                  ส่งออก YAML
                </button>
                
                <button
                  onClick={() => onCategoryAdd?.()}
                  className="btn-premium text-sm"
                >
                  <PlusIcon className="w-4 h-4 mr-1" />
                  เพิ่มหมวดหมู่
                </button>
              </>
            )}
          </div>
        </div>

        {/* Search */}
        {searchable && (
          <div className="relative">
            <SearchIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
            <input
              type="text"
              placeholder="ค้นหาหมวดหมู่..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="input-premium pl-10"
            />
          </div>
        )}
      </div>

      {/* Tree */}
      <div className="p-4 max-h-96 overflow-y-auto">
        <DragDropContext onDragEnd={handleDragEnd}>
          <Droppable droppableId="root" type="CATEGORY">
            {(provided) => (
              <div ref={provided.innerRef} {...provided.droppableProps}>
                {filteredCategories.map((category, index) => (
                  <TreeNode
                    key={category.id}
                    category={category}
                    level={0}
                    index={index}
                    isExpanded={expandedNodes.has(category.id)}
                    isSelected={selectedCategory?.id === category.id}
                    searchTerm={searchTerm}
                    onToggle={handleToggle}
                    onSelect={handleSelect}
                    onEdit={onCategoryEdit!}
                    onDelete={onCategoryDelete!}
                    onAddChild={onCategoryAdd!}
                    onGenerateCode={handleGenerateCode}
                    editable={editable}
                  />
                ))}
                {provided.placeholder}
              </div>
            )}
          </Droppable>
        </DragDropContext>
      </div>

      {/* Import Modal */}
      {showImportModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 w-full max-w-2xl mx-4">
            <h3 className="text-lg font-semibold mb-4">นำเข้าข้อมูล YAML</h3>
            
            <textarea
              value={importYaml}
              onChange={(e) => setImportYaml(e.target.value)}
              placeholder="วางข้อมูล YAML ที่นี่..."
              className="w-full h-64 p-3 border border-gray-300 rounded-lg font-mono text-sm"
            />
            
            <div className="flex justify-end space-x-2 mt-4">
              <button
                onClick={() => setShowImportModal(false)}
                className="btn-secondary"
              >
                ยกเลิก
              </button>
              <button
                onClick={handleImport}
                className="btn-premium"
              >
                นำเข้าข้อมูล
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
