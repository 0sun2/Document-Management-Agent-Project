import { Upload, FileText } from 'lucide-react';
import { useState, useCallback } from 'react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Document } from '@/types';
import { toast } from 'sonner';

interface DocumentUploadProps {
  onUpload: (files: File[]) => void;
}

export function DocumentUpload({ onUpload }: DocumentUploadProps) {
  const [isDragging, setIsDragging] = useState(false);

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const handleDragIn = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  }, []);

  const handleDragOut = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    const files = Array.from(e.dataTransfer.files);
    const validFiles = files.filter(file => {
      const validTypes = ['pdf', 'docx', 'txt', 'md', 'csv', 'pptx'];
      const ext = file.name.split('.').pop()?.toLowerCase();
      return ext && validTypes.includes(ext);
    });

    if (validFiles.length === 0) {
      toast.error('Please upload valid document files (PDF, DOCX, TXT, MD, CSV, PPTX)');
      return;
    }

    onUpload(validFiles);
  }, [onUpload]);

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const files = Array.from(e.target.files);
      onUpload(files);
    }
  };

  return (
    <Card
      className={`p-8 border-2 border-dashed transition-colors ${
        isDragging ? 'border-primary bg-accent' : 'border-border'
      }`}
      onDragEnter={handleDragIn}
      onDragLeave={handleDragOut}
      onDragOver={handleDrag}
      onDrop={handleDrop}
    >
      <div className="flex flex-col items-center justify-center gap-4 text-center">
        <div className="rounded-full bg-primary/10 p-4">
          <Upload className="h-8 w-8 text-primary" />
        </div>
        <div>
          <h3 className="font-semibold text-lg mb-1">문서 업로드</h3>
          <p className="text-sm text-muted-foreground mb-4">
            드래그 & 드롭으로 파일을 추가하거나 클릭하여 선택
          </p>
          <p className="text-xs text-muted-foreground">
            지원 형식: PDF, DOCX, PPTX, TXT, MD, CSV
          </p>
        </div>
        <label htmlFor="file-upload">
          <Button variant="outline" asChild>
            <span>
              <FileText className="h-4 w-4 mr-2" />
              파일 선택
            </span>
          </Button>
        </label>
        <input
          id="file-upload"
          type="file"
          multiple
          accept=".pdf,.docx,.pptx,.txt,.md,.csv"
          onChange={handleFileInput}
          className="hidden"
        />
      </div>
    </Card>
  );
}
