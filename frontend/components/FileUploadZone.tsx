'use client'

import { useCallback, useState, useRef } from 'react'
import { Upload, X, FileText, Loader2 } from 'lucide-react'
import { Button } from '@/components/ui/button'

export interface UploadedFile {
  file: File
  name: string
  size: number
  type: string
  progress: number
  status: 'pending' | 'uploading' | 'done' | 'error'
  errorMessage?: string
}

interface FileUploadZoneProps {
  files: UploadedFile[]
  onFilesChange: (files: UploadedFile[]) => void
  maxFiles?: number
  maxSizeMB?: number
  accept?: string
  label?: string
  description?: string
}

function formatFileSize(bytes: number) {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

export default function FileUploadZone({
  files,
  onFilesChange,
  maxFiles = 5,
  maxSizeMB = 10,
  accept = '.pdf,.jpg,.jpeg,.png',
  label = 'Upload Documents',
  description = 'PDF, JPG, or PNG up to 10 MB each',
}: FileUploadZoneProps) {
  const [dragOver, setDragOver] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  const handleFiles = useCallback(
    (incoming: FileList | null) => {
      if (!incoming) return
      const newFiles: UploadedFile[] = []
      const maxSize = maxSizeMB * 1024 * 1024

      for (let i = 0; i < incoming.length; i++) {
        if (files.length + newFiles.length >= maxFiles) break
        const f = incoming[i]
        if (f.size > maxSize) {
          newFiles.push({
            file: f,
            name: f.name,
            size: f.size,
            type: f.type,
            progress: 0,
            status: 'error',
            errorMessage: `File exceeds ${maxSizeMB} MB limit`,
          })
        } else {
          newFiles.push({
            file: f,
            name: f.name,
            size: f.size,
            type: f.type,
            progress: 0,
            status: 'pending',
          })
        }
      }

      onFilesChange([...files, ...newFiles])
    },
    [files, onFilesChange, maxFiles, maxSizeMB],
  )

  const removeFile = useCallback(
    (idx: number) => {
      onFilesChange(files.filter((_, i) => i !== idx))
    },
    [files, onFilesChange],
  )

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      setDragOver(false)
      handleFiles(e.dataTransfer.files)
    },
    [handleFiles],
  )

  return (
    <div className="space-y-3">
      <p className="text-sm font-medium text-foreground">{label}</p>

      {/* Drop zone */}
      <div
        onDragOver={(e) => {
          e.preventDefault()
          setDragOver(true)
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') inputRef.current?.click()
        }}
        className={`border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition-colors ${
          dragOver
            ? 'border-primary bg-primary/5'
            : 'border-border hover:border-muted-foreground'
        }`}
      >
        <input
          ref={inputRef}
          type="file"
          accept={accept}
          multiple
          className="hidden"
          onChange={(e) => handleFiles(e.target.files)}
        />
        <Upload className="w-8 h-8 mx-auto text-muted-foreground mb-2" />
        <p className="text-sm text-foreground">
          Drag and drop files here, or click to browse
        </p>
        <p className="text-xs text-muted-foreground mt-1">{description}</p>
      </div>

      {/* File list */}
      {files.length > 0 && (
        <ul className="space-y-2">
          {files.map((f, idx) => (
            <li
              key={`${f.name}-${idx}`}
              className="flex items-center gap-3 bg-background border border-border rounded-lg p-3"
            >
              <FileText className="w-5 h-5 text-muted-foreground shrink-0" />
              <div className="flex-1 min-w-0">
                <p className="text-sm text-foreground truncate">{f.name}</p>
                <p className="text-xs text-muted-foreground">
                  {formatFileSize(f.size)}
                  {f.status === 'error' && f.errorMessage && (
                    <span className="text-destructive-foreground ml-2">
                      {f.errorMessage}
                    </span>
                  )}
                </p>
                {f.status === 'uploading' && (
                  <div className="mt-1 h-1 bg-border rounded-full overflow-hidden">
                    <div
                      className="h-full bg-primary rounded-full transition-all"
                      style={{ width: `${f.progress}%` }}
                    />
                  </div>
                )}
              </div>
              {f.status === 'uploading' ? (
                <Loader2 className="w-4 h-4 text-primary animate-spin shrink-0" />
              ) : (
                <Button
                  type="button"
                  variant="ghost"
                  size="icon"
                  className="shrink-0 w-7 h-7 text-muted-foreground hover:text-destructive"
                  onClick={(e) => {
                    e.stopPropagation()
                    removeFile(idx)
                  }}
                  aria-label={`Remove ${f.name}`}
                >
                  <X className="w-4 h-4" />
                </Button>
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
