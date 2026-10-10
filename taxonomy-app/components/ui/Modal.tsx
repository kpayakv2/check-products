'use client'

import React, { useEffect, useRef, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { XIcon } from 'lucide-react'

export const FOCUS_RING =
  'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:ring-offset-2'

export const FOCUSABLE_SELECTOR =
  'button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), a[href], [tabindex]:not([tabindex="-1"])'

export interface ModalProps {
  isOpen: boolean
  onClose: () => void
  title?: React.ReactNode
  titleId?: string
  ariaLabel?: string
  children: React.ReactNode
  className?: string
  maxWidth?: string
  initialFocusRef?: React.RefObject<HTMLElement | null>
  restoreFocusRef?: React.RefObject<HTMLElement | null>
  showCloseButton?: boolean
  closeButtonLabel?: string
  dataTestId?: string
}

export default function Modal({
  isOpen,
  onClose,
  title,
  titleId = 'modal-title',
  ariaLabel,
  children,
  className = '',
  maxWidth = 'max-w-2xl',
  initialFocusRef,
  restoreFocusRef,
  showCloseButton = true,
  closeButtonLabel = 'ปิดหน้าต่าง',
  dataTestId,
}: ModalProps) {
  const dialogRef = useRef<HTMLDivElement | null>(null)
  const previousActiveElementRef = useRef<HTMLElement | null>(null)

  // Save the trigger element before opening to restore focus later
  useEffect(() => {
    if (isOpen) {
      previousActiveElementRef.current = document.activeElement as HTMLElement | null
    }
  }, [isOpen])

  // Focus trap & Escape key listener
  useEffect(() => {
    if (!isOpen) return

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault()
        onClose()
        return
      }

      if (event.key !== 'Tab' || !dialogRef.current) return

      const focusableNodes = Array.from(
        dialogRef.current.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR)
      )

      if (focusableNodes.length === 0) {
        event.preventDefault()
        return
      }

      const firstNode = focusableNodes[0]
      const lastNode = focusableNodes[focusableNodes.length - 1]
      const activeElement = document.activeElement

      if (event.shiftKey && (activeElement === firstNode || !dialogRef.current.contains(activeElement))) {
        event.preventDefault()
        lastNode.focus()
      } else if (!event.shiftKey && activeElement === lastNode) {
        event.preventDefault()
        firstNode.focus()
      }
    }

    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [isOpen, onClose])

  // Auto focus initial element on open
  useEffect(() => {
    if (!isOpen) return

    const timer = setTimeout(() => {
      if (initialFocusRef?.current) {
        initialFocusRef.current.focus()
      } else if (dialogRef.current) {
        const firstFocusable = dialogRef.current.querySelector<HTMLElement>(FOCUSABLE_SELECTOR)
        if (firstFocusable) {
          firstFocusable.focus()
        } else {
          dialogRef.current.focus()
        }
      }
    }, 50)

    return () => clearTimeout(timer)
  }, [isOpen, initialFocusRef])

  // Restore focus on close
  const handleExitComplete = useCallback(() => {
    if (restoreFocusRef?.current) {
      restoreFocusRef.current.focus()
    } else if (previousActiveElementRef.current && typeof previousActiveElementRef.current.focus === 'function') {
      previousActiveElementRef.current.focus()
    }
  }, [restoreFocusRef])

  return (
    <AnimatePresence onExitComplete={handleExitComplete}>
      {isOpen && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-6 bg-slate-900/60 backdrop-blur-md overflow-y-auto"
          onClick={onClose}
        >
          <div
            ref={dialogRef}
            role="dialog"
            aria-modal="true"
            aria-labelledby={title ? titleId : undefined}
            aria-label={ariaLabel}
            data-testid={dataTestId}
            tabIndex={-1}
            onClick={(e) => e.stopPropagation()}
            className={`w-full ${maxWidth} bg-white rounded-3xl shadow-2xl border border-slate-100 overflow-hidden thai-text my-auto max-h-[90vh] flex flex-col focus:outline-none ${className}`}
          >
            <motion.div
              initial={{ opacity: 0, scale: 0.95, y: 15 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95, y: 15 }}
              transition={{ duration: 0.2 }}
              className="flex flex-col h-full w-full overflow-hidden"
            >
              {(title || showCloseButton) && (
                <div className="flex items-center justify-between p-6 border-b border-slate-100 bg-white sticky top-0 z-10">
                  {title && (
                    <div id={titleId} className="text-xl font-black text-slate-800 tracking-tight">
                      {title}
                    </div>
                  )}
                  {!title && <div />}
                  {showCloseButton && (
                    <button
                      type="button"
                      onClick={onClose}
                      aria-label={closeButtonLabel}
                      className={`p-2 text-slate-400 hover:text-slate-700 hover:bg-slate-100 rounded-xl transition-all ${FOCUS_RING}`}
                    >
                      <XIcon className="w-5 h-5" />
                    </button>
                  )}
                </div>
              )}
              <div className="p-6 overflow-y-auto flex-1">{children}</div>
            </motion.div>
          </div>
        </div>
      )}
    </AnimatePresence>
  )
}
