'use client'

import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { toast } from 'react-hot-toast'
import Sidebar from '@/components/Layout/Sidebar'
import Header from '@/components/Layout/Header'
import { 
  SettingsIcon,
  CodeIcon,
  SearchIcon,
  SaveIcon,
  RefreshCwIcon,
  AlertCircleIcon,
  CheckCircleIcon,
  PlusIcon,
  Trash2Icon,
  EditIcon,
  ActivityIcon,
  DownloadIcon,
  UploadIcon,
  ZapIcon,
  ShieldCheckIcon,
  DatabaseIcon,
  XIcon
} from 'lucide-react'
import { DatabaseService } from '@/utils/supabase'

interface RegexRule {
  id: string
  name: string
  description: string
  pattern: string
  flags: string
  category_id: string
  priority: number
  is_active: boolean
  test_cases: string[]
}

interface KeywordRule {
  id: string
  name: string
  description: string
  keywords: string[]
  category_id: string
  match_type: 'contains' | 'exact' | 'fuzzy'
  priority: number
  confidence_score: number
  is_active: boolean
}

interface SystemSettings {
  search: {
    vectorSearchEnabled: boolean
    textSearchEnabled: boolean
    hybridSearchEnabled: boolean
    defaultSearchType: 'vector' | 'text' | 'hybrid'
    maxResults: number
    confidenceThreshold: number
  }
  processing: {
    batchSize: number
    maxConcurrentJobs: number
    retryAttempts: number
    timeoutSeconds: number
  }
  ai: {
    embeddingModel: string
    apiProvider: 'openai' | 'huggingface' | 'local'
    maxTokens: number
    temperature: number
  }
  ui: {
    theme: 'light' | 'dark' | 'auto'
    language: 'th' | 'en' | 'auto'
    itemsPerPage: number
    enableAnimations: boolean
  }
}

export default function SettingsPage() {
  const [activeTab, setActiveTab] = useState<'regex' | 'keywords' | 'system' | 'import'>('regex')
  const [regexRules, setRegexRules] = useState<RegexRule[]>([])
  const [keywordRules, setKeywordRules] = useState<KeywordRule[]>([])
  const [systemSettings, setSystemSettings] = useState<SystemSettings>({
    search: {
      vectorSearchEnabled: true,
      textSearchEnabled: true,
      hybridSearchEnabled: true,
      defaultSearchType: 'hybrid',
      maxResults: 50,
      confidenceThreshold: 0.5
    },
    processing: {
      batchSize: 100,
      maxConcurrentJobs: 5,
      retryAttempts: 3,
      timeoutSeconds: 30
    },
    ai: {
      embeddingModel: 'text-embedding-ada-002',
      apiProvider: 'openai',
      maxTokens: 4000,
      temperature: 0.1
    },
    ui: {
      theme: 'light',
      language: 'th',
      itemsPerPage: 20,
      enableAnimations: true
    }
  })

  const [selectedRule, setSelectedRule] = useState<RegexRule | KeywordRule | null>(null)
  const [showRuleEditor, setShowRuleEditor] = useState(false)
  const [showJsonEditor, setShowJsonEditor] = useState(false)
  const [jsonEditorContent, setJsonEditorContent] = useState('')
  const [testInput, setTestInput] = useState('')
  const [testResults, setTestResults] = useState<any[]>([])
  const [isLoading, setIsLoading] = useState(false)

  // Load data on mount
  useEffect(() => {
    loadRules()
    loadSystemSettings()
  }, [])

  const loadRules = async () => {
    try {
      setIsLoading(true)
      // Load regex and keyword rules from database
      const [regexData, keywordData] = await Promise.all([
        DatabaseService.getRegexRules(),
        DatabaseService.getKeywordRules()
      ])
      setRegexRules(regexData || [])
      setKeywordRules(keywordData || [])
    } catch (error) {
      toast.error('Failed to load rules')
      console.error(error)
    } finally {
      setIsLoading(false)
    }
  }

  const loadSystemSettings = async () => {
    try {
      const settings = await DatabaseService.getSystemSettings()
      if (settings) {
        setSystemSettings(settings)
      }
    } catch (error) {
      console.error('Failed to load system settings:', error)
    }
  }

  const saveSystemSettings = async () => {
    try {
      await DatabaseService.updateSystemSettings(systemSettings)
      toast.success('System settings saved')
    } catch (error) {
      toast.error('Failed to save system settings')
      console.error(error)
    }
  }

  const testRegexRule = (rule: RegexRule, input: string) => {
    try {
      const regex = new RegExp(rule.pattern, rule.flags)
      const matches = input.match(regex)
      return {
        matches: matches || [],
        isMatch: !!matches,
        error: null
      }
    } catch (error) {
      return {
        matches: [],
        isMatch: false,
        error: (error as Error).message
      }
    }
  }

  const testKeywordRule = (rule: KeywordRule, input: string) => {
    const lowerInput = input.toLowerCase()
    const matchedKeywords: string[] = []

    rule.keywords.forEach(keyword => {
      const lowerKeyword = keyword.toLowerCase()
      let isMatch = false

      switch (rule.match_type) {
        case 'exact':
          isMatch = lowerInput === lowerKeyword
          break
        case 'contains':
          isMatch = lowerInput.includes(lowerKeyword)
          break
        case 'fuzzy':
          // Simple fuzzy matching (Levenshtein distance)
          const distance = levenshteinDistance(lowerInput, lowerKeyword)
          isMatch = distance <= Math.max(1, Math.floor(lowerKeyword.length * 0.2))
          break
      }

      if (isMatch) {
        matchedKeywords.push(keyword)
      }
    })

    return {
      matchedKeywords,
      isMatch: matchedKeywords.length > 0,
      confidence: matchedKeywords.length / rule.keywords.length
    }
  }

  const levenshteinDistance = (str1: string, str2: string): number => {
    const matrix = Array(str2.length + 1).fill(null).map(() => Array(str1.length + 1).fill(null))

    for (let i = 0; i <= str1.length; i++) matrix[0][i] = i
    for (let j = 0; j <= str2.length; j++) matrix[j][0] = j

    for (let j = 1; j <= str2.length; j++) {
      for (let i = 1; i <= str1.length; i++) {
        const indicator = str1[i - 1] === str2[j - 1] ? 0 : 1
        matrix[j][i] = Math.min(
          matrix[j][i - 1] + 1,
          matrix[j - 1][i] + 1,
          matrix[j - 1][i - 1] + indicator
        )
      }
    }

    return matrix[str2.length][str1.length]
  }

  const runTests = () => {
    if (!testInput.trim()) {
      toast.error('Please enter test input')
      return
    }

    const results: any[] = []

    if (activeTab === 'regex') {
      regexRules.forEach(rule => {
        if (rule.is_active) {
          const result = testRegexRule(rule, testInput)
          results.push({
            rule,
            type: 'regex',
            ...result
          })
        }
      })
    } else if (activeTab === 'keywords') {
      keywordRules.forEach(rule => {
        if (rule.is_active) {
          const result = testKeywordRule(rule, testInput)
          results.push({
            rule,
            type: 'keyword',
            ...result
          })
        }
      })
    }

    setTestResults(results)
  }

  const exportRules = () => {
    const data = {
      regexRules,
      keywordRules,
      systemSettings,
      exportedAt: new Date().toISOString()
    }

    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `taxonomy-rules-${new Date().toISOString().split('T')[0]}.json`
    a.click()
    URL.revokeObjectURL(url)
    toast.success('Rules exported successfully')
  }

  const importRules = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return

    const reader = new FileReader()
    reader.onload = (e) => {
      try {
        const data = JSON.parse(e.target?.result as string)
        
        if (data.regexRules) setRegexRules(data.regexRules)
        if (data.keywordRules) setKeywordRules(data.keywordRules)
        if (data.systemSettings) setSystemSettings(data.systemSettings)
        
        toast.success('Rules imported successfully')
      } catch (error) {
        toast.error('Failed to import rules: Invalid JSON format')
      }
    }
    reader.readAsText(file)
  }

  const openJsonEditor = (type: 'regex' | 'keywords' | 'system') => {
    let content = ''
    switch (type) {
      case 'regex':
        content = JSON.stringify(regexRules, null, 2)
        break
      case 'keywords':
        content = JSON.stringify(keywordRules, null, 2)
        break
      case 'system':
        content = JSON.stringify(systemSettings, null, 2)
        break
    }
    setJsonEditorContent(content)
    setShowJsonEditor(true)
  }

  const saveJsonEditor = () => {
    try {
      const data = JSON.parse(jsonEditorContent)
      
      if (activeTab === 'regex') {
        setRegexRules(data)
      } else if (activeTab === 'keywords') {
        setKeywordRules(data)
      } else if (activeTab === 'system') {
        setSystemSettings(data)
      }
      
      setShowJsonEditor(false)
      toast.success('JSON updated successfully')
    } catch (error) {
      toast.error('Invalid JSON format')
    }
  }

  const tabs = [
    { id: 'regex', name: 'Regex Rules', icon: SearchIcon },
    { id: 'keywords', name: 'Keyword Rules', icon: CodeIcon },
    { id: 'system', name: 'System Settings', icon: SettingsIcon },
    { id: 'import', name: 'Import/Export', icon: DownloadIcon }
  ]

  return (
    <div className="flex h-screen bg-gray-50 font-sans">
      <Sidebar />
      
      <div className="flex-1 flex flex-col overflow-hidden">
        <Header />
        
        <main className="flex-1 overflow-x-hidden overflow-y-auto p-8 relative">
          {/* Decorative Background Elements */}
          <div className="absolute top-0 right-0 w-[500px] h-[500px] bg-indigo-50/50 rounded-full blur-[120px] -mr-48 -mt-48 pointer-events-none" />
          <div className="absolute bottom-10 left-10 w-[300px] h-[300px] bg-emerald-50/50 rounded-full blur-[100px] pointer-events-none" />

          <div className="max-w-7xl mx-auto relative z-10">
            {/* Page Header */}
            <div className="mb-10">
              <h1 className="text-4xl font-extrabold text-slate-900 tracking-tight thai-text uppercase">
                System Workflow Configuration
              </h1>
              <p className="mt-2 text-slate-500 font-medium thai-text">
                จัดการกฎเกณฑ์ของ AI, กติกาการจัดหมวดหมู่ และการตั้งค่าเชิงลึกของระบบจัดการสินค้า
              </p>
            </div>

            {/* Tab Navigation */}
            <div className="mb-10 p-1.5 bg-slate-100 rounded-3xl inline-flex gap-1 border border-slate-200/60 shadow-inner">
              {tabs.map((tab) => {
                const Icon = tab.icon
                const isActive = activeTab === tab.id
                return (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id as any)}
                    className={`
                      relative flex items-center px-6 py-3 rounded-2xl text-xs font-black uppercase tracking-widest transition-all duration-300
                      ${isActive
                        ? 'text-indigo-600'
                        : 'text-slate-400 hover:text-slate-600'
                      }
                    `}
                  >
                    {isActive && (
                      <motion.div
                        layoutId="active-tab-indicator"
                        className="absolute inset-0 bg-white rounded-2xl shadow-sm border border-indigo-100/50"
                        transition={{ type: "spring", bounce: 0.2, duration: 0.6 }}
                      />
                    )}
                    <span className="relative z-10 flex items-center">
                          <span className={`text-xs font-bold uppercase tracking-wider transition-colors ${activeTab === tab.id ? 'text-indigo-600' : 'text-slate-400'}`}>
                            {tab.name}
                          </span>
                    </span>
                  </button>
                )
              })}
            </div>

            {/* Content Container */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 items-start">
              {/* Dynamic Content Panel */}
              <div className="lg:col-span-2">
                <AnimatePresence mode="wait">
                  <motion.div
                    key={activeTab}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -10 }}
                    transition={{ duration: 0.3 }}
                    className="premium-card p-10 bg-white/60 backdrop-blur-xl border border-white"
                  >
                    {/* Regex Rules Tab */}
                    {activeTab === 'regex' && (
                      <div className="space-y-8">
                        <div className="flex items-center justify-between">
                          <div>
                            <h2 className="text-2xl font-black text-slate-800 tracking-tight uppercase">Regular Expression Engine</h2>
                            <p className="text-sm font-medium text-slate-400 thai-text">กำหนดกฎสำหรับการแยกคุณลักษณะสินค้าจากข้อความอัตโนมัติ</p>
                          </div>
                          <div className="hidden sm:flex space-x-3">
                            <button
                              onClick={() => openJsonEditor('regex')}
                              className="btn-premium-secondary"
                            >
                              <CodeIcon className="w-4 h-4 mr-2" />
                              JSON Setup
                            </button>
                            <button className="btn-premium px-8">
                              <PlusIcon className="w-4 h-4 mr-2" />
                              Add Identifier
                            </button>
                          </div>
                        </div>

                        <div className="grid grid-cols-1 gap-6">
                          {regexRules.map((rule, idx) => (
                            <div 
                              key={rule.id}
                              className="bg-white/80 rounded-[32px] p-8 border border-slate-100 group hover:border-indigo-100 transition-all duration-300 shadow-sm hover:shadow-xl hover:shadow-indigo-100/10"
                            >
                              <div className="flex items-start justify-between gap-6">
                                <div className="flex-1">
                                  <div className="flex items-center space-x-4 mb-3">
                                    <div className="w-10 h-10 bg-indigo-50 rounded-xl flex items-center justify-center text-indigo-600 border border-indigo-100/50 group-hover:scale-110 transition-transform">
                                      <SearchIcon className="w-5 h-5" />
                                    </div>
                                    <h3 className="text-xl font-extrabold text-slate-800 tracking-tight">{rule.name}</h3>
                                    <span className={`px-3 py-1 rounded-full text-[10px] font-black uppercase tracking-wider ${
                                      rule.is_active ? 'bg-emerald-50 text-emerald-600 border border-emerald-100' : 'bg-slate-50 text-slate-400 border border-slate-100'
                                    }`}>
                                      {rule.is_active ? 'Production Ready' : 'DRAFT'}
                                    </span>
                                  </div>
                                  <p className="text-sm font-medium text-slate-500 thai-text leading-relaxed mb-6">{rule.description}</p>
                                  
                                  <div className="flex flex-wrap gap-4 items-center">
                                    <div className="px-4 py-2 bg-slate-900 rounded-2xl flex items-center gap-3 border border-white/10 shadow-lg">
                                      <span className="text-[10px] font-black text-indigo-400 uppercase tracking-widest border-r border-white/10 pr-3">Pattern</span>
                                      <code className="text-xs font-mono font-bold text-indigo-100">{rule.pattern}</code>
                                    </div>
                                  </div>
                                </div>
                                
                                <div className="flex flex-col gap-2">
                                  <button onClick={() => { setSelectedRule(rule); setShowRuleEditor(true); }} className="p-3 bg-slate-50 rounded-xl text-slate-400 hover:text-indigo-600 hover:bg-white border border-slate-100 hover:border-indigo-100 transition-all">
                                    <EditIcon className="w-4 h-4" />
                                  </button>
                                  <button className="p-3 bg-slate-50 rounded-xl text-slate-400 hover:text-rose-600 hover:bg-rose-50 border border-slate-100 hover:border-rose-100 transition-all">
                                    <Trash2Icon className="w-4 h-4" />
                                  </button>
                                </div>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Keyword Rules Tab */}
                    {activeTab === 'keywords' && (
                      <div className="space-y-8">
                         <div className="flex items-center justify-between">
                          <div>
                            <h2 className="text-2xl font-black text-slate-800 tracking-tight uppercase">Keyword Logic</h2>
                            <p className="text-sm font-medium text-slate-400 thai-text">ตรรกะสำหรับการจับคูสินค้าด้วยคำสำคัญ (Keyword Mapping)</p>
                          </div>
                          <div className="hidden sm:flex space-x-3">
                            <button
                              onClick={() => openJsonEditor('keywords')}
                              className="btn-premium-secondary"
                            >
                              <CodeIcon className="w-4 h-4 mr-2" />
                              JSON Setup
                            </button>
                            <button className="btn-premium px-8">
                              <PlusIcon className="w-4 h-4 mr-2" />
                              Add Keywords
                            </button>
                          </div>
                        </div>

                        <div className="grid grid-cols-1 gap-6">
                          {keywordRules.map((rule) => (
                            <div 
                              key={rule.id}
                              className="bg-white/80 rounded-[32px] p-8 border border-slate-100 group hover:border-indigo-100 transition-all duration-300 shadow-sm hover:shadow-xl hover:shadow-indigo-100/10"
                            >
                              <div className="flex items-start justify-between gap-6">
                                <div className="flex-1">
                                  <div className="flex items-center space-x-4 mb-3">
                                    <div className="w-10 h-10 bg-emerald-50 rounded-xl flex items-center justify-center text-emerald-600 border border-emerald-100/50 group-hover:scale-110 transition-transform">
                                      <CodeIcon className="w-5 h-5" />
                                    </div>
                                    <h3 className="text-xl font-extrabold text-slate-800 tracking-tight">{rule.name}</h3>
                                    <span className="px-3 py-1 rounded-full text-[10px] font-black uppercase tracking-wider bg-indigo-50 text-indigo-600 border border-indigo-100">
                                      {rule.match_type} MATCH
                                    </span>
                                  </div>
                                  <p className="text-sm font-medium text-slate-500 thai-text leading-relaxed mb-6">{rule.description}</p>
                                  
                                  <div className="flex flex-wrap gap-2">
                                    {rule.keywords.slice(0, 10).map((k, i) => (
                                      <span key={i} className="px-3 py-1 bg-slate-50 border border-slate-100 rounded-lg text-xs font-bold text-slate-600">{k}</span>
                                    ))}
                                    {rule.keywords.length > 10 && (
                                      <span className="text-xs font-bold text-slate-400">+{rule.keywords.length - 10} more</span>
                                    )}
                                  </div>
                                </div>
                                <div className="flex flex-col gap-2">
                                  <button className="p-3 bg-slate-50 rounded-xl text-slate-400 hover:text-indigo-600 hover:bg-white border border-slate-100 hover:border-indigo-100 transition-all">
                                    <EditIcon className="w-4 h-4" />
                                  </button>
                                  <button className="p-3 bg-slate-50 rounded-xl text-slate-400 hover:text-rose-600 hover:bg-rose-50 border border-slate-100 hover:border-rose-100 transition-all">
                                    <Trash2Icon className="w-4 h-4" />
                                  </button>
                                </div>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* System Settings Tab */}
                    {activeTab === 'system' && (
                      <div className="space-y-12">
                        <div className="flex items-center justify-between">
                          <div>
                            <h2 className="text-2xl font-black text-slate-800 tracking-tight uppercase">Global Policy</h2>
                            <p className="text-sm font-medium text-slate-400 thai-text">คอนฟิกูเรชันหลักสำหรับ AI Engine และ System Behavior</p>
                          </div>
                          <button onClick={saveSystemSettings} className="btn-premium px-10">
                            <SaveIcon className="w-4 h-4 mr-2" />
                            Apply Changes
                          </button>
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-10">
                          <section className="space-y-6">
                            <div className="flex items-center gap-3">
                               <div className="p-2 bg-indigo-50 rounded-xl text-indigo-600 border border-indigo-100/50">
                                  <SearchIcon className="w-5 h-5" />
                               </div>
                               <h3 className="text-lg font-black text-slate-900 tracking-tight uppercase">Search & Indexing</h3>
                            </div>
                            
                            <div className="space-y-4">
                              <div>
                                <label className="block text-[10px] font-black text-slate-400 uppercase tracking-widest mb-2 px-1">Default Search Strategy</label>
                                <select
                                  value={systemSettings.search.defaultSearchType}
                                  onChange={(e) => setSystemSettings(prev => ({
                                    ...prev,
                                    search: { ...prev.search, defaultSearchType: e.target.value as any }
                                  }))}
                                  className="select-premium w-full text-sm font-black"
                                >
                                  <option value="vector">Semantic Vector Search</option>
                                  <option value="text">Fuzzy Text Matching</option>
                                  <option value="hybrid">Intelligent Hybrid (AI Recommended)</option>
                                </select>
                              </div>
                              
                              <div className="grid grid-cols-2 gap-4">
                                <div>
                                  <label className="block text-[10px] font-black text-slate-400 uppercase tracking-widest mb-2 px-1">Max Result Cap</label>
                                  <input
                                    type="number"
                                    value={systemSettings.search.maxResults}
                                    onChange={(e) => setSystemSettings(prev => ({
                                      ...prev,
                                      search: { ...prev.search, maxResults: parseInt(e.target.value) }
                                    }))}
                                    className="input-premium w-full text-sm font-black"
                                  />
                                </div>
                                <div>
                                  <label className="block text-[10px] font-black text-slate-400 uppercase tracking-widest mb-2 px-1">Confidence Thres.</label>
                                  <input
                                    type="number" step="0.05"
                                    value={systemSettings.search.confidenceThreshold}
                                    onChange={(e) => setSystemSettings(prev => ({
                                      ...prev,
                                      search: { ...prev.search, confidenceThreshold: parseFloat(e.target.value) }
                                    }))}
                                    className="input-premium w-full text-sm font-black"
                                  />
                                </div>
                              </div>
                            </div>
                          </section>

                          <section className="space-y-6">
                            <div className="flex items-center gap-3">
                               <div className="p-2 bg-emerald-50 rounded-xl text-emerald-600 border border-emerald-100/50">
                                  <ZapIcon className="w-5 h-5" />
                               </div>
                               <h3 className="text-lg font-black text-slate-900 tracking-tight uppercase">AI Engine Settings</h3>
                            </div>

                            <div className="space-y-4">
                              <div>
                                <label className="block text-[10px] font-black text-slate-400 uppercase tracking-widest mb-2 px-1">Active Model</label>
                                <select
                                  value={systemSettings.ai.embeddingModel}
                                  onChange={(e) => setSystemSettings(prev => ({
                                    ...prev,
                                    ai: { ...prev.ai, embeddingModel: e.target.value }
                                  }))}
                                  className="select-premium w-full text-sm font-black"
                                >
                                  <option value="text-embedding-ada-002">OpenAI Ada-V2</option>
                                  <option value="multilingual-e5-large">Multilingual E5 Large</option>
                                </select>
                              </div>
                              
                              <div>
                                <label className="block text-[10px] font-black text-slate-400 uppercase tracking-widest mb-2 px-1">Preference Provider</label>
                                <div className="grid grid-cols-3 gap-2">
                                  {['openai', 'huggingface', 'local'].map((provider) => (
                                    <button
                                      key={provider}
                                      onClick={() => setSystemSettings(prev => ({
                                        ...prev,
                                        ai: { ...prev.ai, apiProvider: provider as any }
                                      }))}
                                      className={`px-3 py-2 rounded-xl text-[9px] font-black uppercase tracking-widest border transition-all ${
                                        systemSettings.ai.apiProvider === provider 
                                          ? 'bg-indigo-600 text-white border-indigo-600 shadow-lg shadow-indigo-600/20' 
                                          : 'bg-white text-slate-400 border-slate-100 hover:border-indigo-100 hover:text-slate-600'
                                      }`}
                                    >
                                      {provider}
                                    </button>
                                  ))}
                                </div>
                              </div>
                            </div>
                          </section>
                        </div>
                      </div>
                    )}

                    {/* Import/Export Tab */}
                    {activeTab === 'import' && (
                      <div className="space-y-12">
                        <div>
                          <h2 className="text-2xl font-black text-slate-800 tracking-tight uppercase mb-2">Workspace Portability</h2>
                          <p className="text-sm font-medium text-slate-400 thai-text">แบ็คอัพหรืออพยพกฎเกณฑ์และความตั้งค่าของระบบ</p>
                        </div>
                        
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-10">
                          <div className="bg-white/80 rounded-[40px] p-8 border border-slate-100 shadow-sm flex flex-col items-center text-center">
                            <div className="w-16 h-16 bg-indigo-50 rounded-3xl flex items-center justify-center text-indigo-600 mb-6 group-hover:scale-110 transition-transform">
                              <DownloadIcon className="w-8 h-8" />
                            </div>
                            <h3 className="text-xl font-bold text-slate-900 tracking-tight uppercase mb-2">Export Library</h3>
                            <p className="text-sm font-medium text-slate-400 thai-text mb-8 px-8">บันทึกกฎทั้งหมดลงไฟล์ JSON สำหรับใช้เป็นชุดข้อมูลสำรองหรือย้ายเครื่อง</p>
                            <button onClick={exportRules} className="btn-premium w-full justify-center py-4">
                              Generate Ruleset Snapshot
                            </button>
                          </div>

                          <div className="bg-white/80 rounded-[40px] p-8 border border-slate-100 shadow-sm flex flex-col items-center text-center">
                            <div className="w-16 h-16 bg-amber-50 rounded-3xl flex items-center justify-center text-amber-600 mb-6 group-hover:scale-110 transition-transform">
                              <UploadIcon className="w-8 h-8" />
                            </div>
                            <h3 className="text-xl font-bold text-slate-900 tracking-tight uppercase mb-2">Sync Infrastructure</h3>
                            <p className="text-sm font-medium text-slate-400 thai-text mb-8 px-8">นำเข้าชุดกฎเกณฑ์จากไฟล์เดิม ระวัง: ข้อมูลปัจจุบันจะถูกเขียนทับทันที</p>
                            <input type="file" accept=".json" onChange={importRules} className="hidden" id="import-file" />
                            <label htmlFor="import-file" className="btn-premium-secondary w-full justify-center py-4 cursor-pointer">
                              Upload Rule Archive
                            </label>
                          </div>
                        </div>
                      </div>
                    )}
                  </motion.div>
                </AnimatePresence>
              </div>

              {/* Side Panel - Rule Tester */}
              <div className="lg:col-span-1">
                <div className="premium-card p-0 overflow-hidden bg-white/40 border-indigo-100/30 sticky top-8">
                  <div className="p-8 border-b border-white">
                    <div className="flex items-center gap-3 mb-6">
                       <div className="w-10 h-10 bg-indigo-600 rounded-xl flex items-center justify-center text-white shadow-lg shadow-indigo-600/20">
                          <ActivityIcon className="w-5 h-5" />
                       </div>
                       <div>
                          <h3 className="text-lg font-black text-slate-900 tracking-tight uppercase">Rule Sandbox</h3>
                          <p className="text-[10px] font-bold text-indigo-400 uppercase tracking-widest">Isolated Environment</p>
                       </div>
                    </div>
                    
                    <div className="space-y-6">
                      <div>
                        <label className="block text-[10px] font-black text-slate-400 uppercase tracking-widest mb-3 px-1">Source Payload (Single Item)</label>
                        <textarea
                          value={testInput}
                          onChange={(e) => setTestInput(e.target.value)}
                          placeholder="วางข้อความชื่อสินค้าสำหรับทดสอบ..."
                          className="input-premium w-full h-32 text-sm font-semibold thai-text leading-relaxed p-4 bg-white/50"
                        />
                      </div>

                      <button
                        onClick={runTests}
                        className="btn-premium w-full py-4 text-sm tracking-[0.1em] justify-center group shadow-indigo-600/25"
                      >
                        <ZapIcon className="w-4 h-4 mr-2 group-hover:animate-pulse" />
                        Execute Core Test
                      </button>
                    </div>
                  </div>

                  {/* Test Results Display */}
                  <div className="p-8 bg-indigo-50/20 min-h-[300px]">
                    <div className="flex items-center justify-between mb-6">
                       <h4 className="text-[10px] font-black text-slate-400 uppercase tracking-[0.2em] px-1">Logic Trace Results</h4>
                       <span className="text-[10px] font-bold text-indigo-500 bg-white px-2 py-0.5 rounded-md border border-indigo-100/50">{testResults.length} Hits</span>
                    </div>

                    {testResults.length > 0 ? (
                      <div className="space-y-4">
                        {testResults.map((result, index) => (
                          <div 
                            key={index}
                            className={`p-5 rounded-3xl border shadow-sm relative overflow-hidden group ${
                              result.isMatch ? 'bg-white border-emerald-100 text-emerald-800' : 'bg-white/40 border-slate-100 text-slate-300 opacity-60'
                            }`}
                          >
                            <div className="flex items-center gap-3 relative z-10">
                              {result.isMatch ? (
                                <CheckCircleIcon className="w-4 h-4 text-emerald-500" />
                              ) : (
                                <XIcon className="w-4 h-4 text-slate-200" />
                              )}
                              <span className="font-extrabold text-xs uppercase tracking-tight">{result.rule.name}</span>
                            </div>

                            {result.isMatch && (
                               <div className="mt-4 flex flex-wrap gap-2 relative z-10">
                                  {(result.matches || result.matchedKeywords || []).map((match: string, mI: number) => (
                                     <span key={mI} className="px-2 py-1 bg-emerald-600 text-white rounded-lg text-[10px] font-black font-mono shadow-sm">
                                        {match}
                                     </span>
                                  ))}
                               </div>
                            )}
                          </div>
                        ))}
                      </div>
                    ) : (
                      <div className="flex flex-col items-center justify-center py-20 text-center">
                         <div className="w-16 h-16 bg-white rounded-full flex items-center justify-center shadow-sm border border-slate-100 mb-6">
                            <ActivityIcon className="w-8 h-8 text-indigo-400" />
                         </div>
                         <p className="text-[10px] font-black text-slate-300 uppercase tracking-widest leading-relaxed px-10">
                            ไม่มีกฎที่จับคู่กันได้ (Miss)<br/>กรุณาเพิ่มความเข้มข้นของ Keywords หรือปรับปรุง Regex Pattern
                         </p>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </main>
      </div>

      {/* JSON Editor Modal */}
      <AnimatePresence>
        {showJsonEditor && (
          <div className="fixed inset-0 bg-slate-900/60 backdrop-blur-md flex items-center justify-center z-[100] p-4 sm:p-10">
            <motion.div 
              initial={{ opacity: 0, scale: 0.9, y: 20 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.9, y: 20 }}
              className="bg-slate-900 rounded-[56px] w-full max-w-5xl shadow-2xl border border-white/10 flex flex-col overflow-hidden max-h-full"
            >
              <div className="p-10 border-b border-white/5 flex items-center justify-between bg-slate-900">
                <div className="flex items-center gap-4">
                  <div className="p-3 bg-indigo-500/20 rounded-2xl text-indigo-400 border border-indigo-500/30">
                    <CodeIcon className="w-6 h-6" />
                  </div>
                  <div>
                    <h3 className="text-2xl font-black text-white tracking-tight uppercase">Infrastructure Direct Override</h3>
                    <p className="text-xs font-medium text-indigo-300/60 uppercase tracking-widest mt-1">Direct System Parameter Modification</p>
                  </div>
                </div>
                
                <button
                  onClick={() => setShowJsonEditor(false)}
                  className="p-4 text-white/50 hover:text-white hover:bg-white/5 rounded-3xl transition-all"
                >
                  <XIcon className="w-6 h-6" />
                </button>
              </div>
              
              <div className="flex-1 p-10 bg-slate-800/20">
                <textarea
                  value={jsonEditorContent}
                  onChange={(e) => setJsonEditorContent(e.target.value)}
                  className="w-full h-full min-h-[400px] bg-transparent font-mono text-sm border-none text-indigo-100 focus:ring-0 resize-none leading-relaxed"
                  placeholder="Insert valid JSON payload here..."
                />
              </div>
              
              <div className="p-10 bg-slate-900 border-t border-white/5 flex flex-col lg:flex-row items-center justify-between gap-6">
                <div className="flex items-start gap-4 p-5 bg-amber-500/10 border border-amber-500/20 rounded-3xl max-w-2xl">
                   <AlertCircleIcon className="w-6 h-6 text-amber-500 shrink-0" />
                   <p className="text-xs font-medium text-amber-200 thai-text leading-relaxed">
                      คำเตือน: การเปลี่ยนโครงสร้าง JSON โดยตรงอาจส่งผลกระทบต่อความเสถียรของระบบ (Parsing Integrity) และอาจทำให้ฟังก์ชันบางส่วนทำงานผิดพลาดได้ กรุณาตรวจสอบ Syntax และ Schema ให้ถูกต้องก่อนกด Commit
                   </p>
                </div>

                <div className="flex items-center gap-4 w-full lg:w-auto">
                  <button
                    onClick={() => setShowJsonEditor(false)}
                    className="flex-1 lg:flex-none px-8 py-4 bg-white/5 hover:bg-white/10 text-white rounded-2xl text-xs font-black uppercase tracking-widest transition-all"
                  >
                    Discard Changes
                  </button>
                  <button
                    onClick={saveJsonEditor}
                    className="flex-1 lg:flex-none btn-premium px-12 py-4"
                  >
                    Commit Parameters
                  </button>
                </div>
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>
    </div>
  )
}

