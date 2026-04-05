'use client'

import React, { useState } from 'react'
import { motion } from 'framer-motion'
import {
  ChevronRightIcon,
  ChevronDownIcon,
  FolderIcon,
  FolderOpenIcon,
  PlusIcon,
  EditIcon,
  TrashIcon
} from 'lucide-react'
import { TaxonomyNode } from '@/utils/supabase'

interface TaxonomyTreeProps {
  categories: TaxonomyNode[]
  onCategorySelect?: (category: TaxonomyNode) => void
  onCategoryAdd?: (parentId?: string) => void
  onCategoryEdit?: (category: TaxonomyNode) => void
  onCategoryDelete?: (category: TaxonomyNode) => void
  selectedCategoryId?: string
  editable?: boolean
}

interface TreeNodeProps {
  category: TaxonomyNode
  level: number
  onSelect?: (category: TaxonomyNode) => void
  onAdd?: (parentId: string) => void
  onEdit?: (category: TaxonomyNode) => void
  onDelete?: (category: TaxonomyNode) => void
  selectedCategoryId?: string
  editable?: boolean
}

const TreeNode: React.FC<TreeNodeProps> = ({ 
  category, 
  level, 
  onSelect, 
  onAdd, 
  onEdit, 
  onDelete, 
  selectedCategoryId,
  editable = false
}) => {
  const [isExpanded, setIsExpanded] = useState(true)
  const hasChildren = category.children && category.children.length > 0
  const isSelected = selectedCategoryId === category.id

  const handleToggle = () => {
    if (hasChildren) {
      setIsExpanded(!isExpanded)
    }
  }

  const handleSelect = () => {
    onSelect?.(category)
  }

  return (
    <div className="select-none">
      <div 
        className={`flex items-center py-2 px-2 rounded-lg cursor-pointer transition-colors ${
          isSelected 
            ? 'bg-blue-100 text-blue-900 border border-blue-200' 
            : 'hover:bg-gray-100'
        }`}
        style={{ paddingLeft: `${level * 20 + 8}px` }}
        onClick={handleSelect}
      >
        {/* Expand/Collapse Button */}
        <button
          onClick={(e) => {
            e.stopPropagation()
            handleToggle()
          }}
          className="mr-2 p-1 rounded hover:bg-gray-200 transition-colors"
        >
          {hasChildren ? (
            isExpanded ? (
              <ChevronDownIcon className="h-4 w-4 text-gray-500" />
            ) : (
              <ChevronRightIcon className="h-4 w-4 text-gray-500" />
            )
          ) : (
            <div className="h-4 w-4" />
          )}
        </button>

        {/* Folder Icon */}
        <div className="mr-3">
          {hasChildren ? (
            isExpanded ? (
              <FolderOpenIcon className="h-5 w-5 text-blue-500" />
            ) : (
              <FolderIcon className="h-5 w-5 text-blue-500" />
            )
          ) : (
            <div className="h-5 w-5 bg-gray-300 rounded-sm" />
          )}
        </div>

        {/* Category Name */}
        <div className="flex-1 min-w-0">
          <div className="font-medium text-gray-900 truncate">
            {category.name_th}
          </div>
          {category.name_en && (
            <div className="text-sm text-gray-500 truncate">
              {category.name_en}
            </div>
          )}
        </div>

        {/* Action Buttons */}
        {editable && (
          <div className="flex items-center gap-1 ml-2">
            <button
              onClick={(e) => {
                e.stopPropagation()
                onAdd?.(category.id)
              }}
              className="p-1 rounded hover:bg-blue-100 text-blue-600 transition-colors"
              title="เพิ่มหมวดหมู่ย่อย"
            >
              <PlusIcon className="h-4 w-4" />
            </button>
            <button
              onClick={(e) => {
                e.stopPropagation()
                onEdit?.(category)
              }}
              className="p-1 rounded hover:bg-yellow-100 text-yellow-600 transition-colors"
              title="แก้ไข"
            >
              <EditIcon className="h-4 w-4" />
            </button>
            <button
              onClick={(e) => {
                e.stopPropagation()
                onDelete?.(category)
              }}
              className="p-1 rounded hover:bg-red-100 text-red-600 transition-colors"
              title="ลบ"
            >
              <TrashIcon className="h-4 w-4" />
            </button>
          </div>
        )}
      </div>

      {/* Children */}
      {hasChildren && isExpanded && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          exit={{ opacity: 0, height: 0 }}
          transition={{ duration: 0.2 }}
        >
          {category.children?.map((child) => (
            <TreeNode
              key={child.id}
              category={child}
              level={level + 1}
              onSelect={onSelect}
              onAdd={onAdd}
              onEdit={onEdit}
              onDelete={onDelete}
              selectedCategoryId={selectedCategoryId}
              editable={editable}
            />
          ))}
        </motion.div>
      )}
    </div>
  )
}

const TaxonomyTree: React.FC<TaxonomyTreeProps> = ({
  categories,
  onCategorySelect,
  onCategoryAdd,
  onCategoryEdit,
  onCategoryDelete,
  selectedCategoryId,
  editable = false
}) => {
  // Build tree structure
  const buildTree = (nodes: TaxonomyNode[]): TaxonomyNode[] => {
    const nodeMap = new Map<string, TaxonomyNode>()
    const rootNodes: TaxonomyNode[] = []

    // Create map and add children array
    nodes.forEach(node => {
      nodeMap.set(node.id, { ...node, children: [] })
    })

    // Build tree structure
    nodes.forEach(node => {
      const nodeWithChildren = nodeMap.get(node.id)!
      if (node.parent_id && nodeMap.has(node.parent_id)) {
        const parent = nodeMap.get(node.parent_id)!
        if (!parent.children) parent.children = []
        parent.children.push(nodeWithChildren)
      } else {
        rootNodes.push(nodeWithChildren)
      }
    })

    // Sort by sort_order
    const sortNodes = (nodes: TaxonomyNode[]): TaxonomyNode[] => {
      return nodes.sort((a, b) => (a.sort_order || 0) - (b.sort_order || 0))
        .map(node => ({
          ...node,
          children: node.children ? sortNodes(node.children) : []
        }))
    }

    return sortNodes(rootNodes)
  }

  const treeData = buildTree(categories)

  if (categories.length === 0) {
    return (
      <div className="text-center py-8 text-gray-500">
        <FolderIcon className="mx-auto h-12 w-12 text-gray-300 mb-4" />
        <p>ไม่มีหมวดหมู่</p>
        {editable && (
          <button
            onClick={() => onCategoryAdd?.()}
            className="mt-4 inline-flex items-center px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            <PlusIcon className="h-4 w-4 mr-2" />
            เพิ่มหมวดหมู่แรก
          </button>
        )}
      </div>
    )
  }

  return (
    <div className="space-y-1">
      {treeData.map((category) => (
        <TreeNode
          key={category.id}
          category={category}
          level={0}
          onSelect={onCategorySelect}
          onAdd={onCategoryAdd}
          onEdit={onCategoryEdit}
          onDelete={onCategoryDelete}
          selectedCategoryId={selectedCategoryId}
          editable={editable}
        />
      ))}
      
      {editable && (
        <div className="pt-4">
          <button
            onClick={() => onCategoryAdd?.()}
            className="inline-flex items-center px-4 py-2 text-blue-600 border border-blue-300 rounded-lg hover:bg-blue-50 transition-colors"
          >
            <PlusIcon className="h-4 w-4 mr-2" />
            เพิ่มหมวดหมู่หลัก
          </button>
        </div>
      )}
    </div>
  )
}

export default TaxonomyTree
