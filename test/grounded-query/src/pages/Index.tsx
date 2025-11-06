import { useState } from 'react';
import { Settings, FileText, PanelLeftClose, PanelLeft } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { ThemeToggle } from '@/components/ThemeToggle';
import { DocumentUpload } from '@/components/DocumentUpload';
import { DocumentList } from '@/components/DocumentList';
import { ChatInterface } from '@/components/ChatInterface';
import { Document } from '@/types';
import { toast } from 'sonner';
import { vectorStore } from '@/adapters/vectorStore';

const Index = () => {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [showDocPanel, setShowDocPanel] = useState(true);

  const handleUpload = async (files: File[]) => {
    const newDocs: Document[] = files.map((file) => ({
      id: Math.random().toString(36).substr(2, 9),
      title: file.name,
      type: file.name.split('.').pop() as any,
      size: file.size,
      uploadedAt: new Date(),
      status: 'pending',
      tags: [],
      progress: 0,
    }));

    setDocuments((prev) => [...prev, ...newDocs]);
    toast.success(`${files.length}개 파일 업로드 시작`);

    // Process each file
    for (let i = 0; i < newDocs.length; i++) {
      const doc = newDocs[i];
      const file = files[i];

      // Update to indexing
      setDocuments((prev) =>
        prev.map((d) => (d.id === doc.id ? { ...d, status: 'indexing' } : d))
      );

      // Simulate progress
      for (let progress = 0; progress <= 90; progress += 20) {
        await new Promise((resolve) => setTimeout(resolve, 200));
        setDocuments((prev) =>
          prev.map((d) => (d.id === doc.id ? { ...d, progress } : d))
        );
      }

      try {
        // Upload to FastAPI backend (handles embedding automatically)
        await vectorStore.indexDocument(doc.id, [], doc.title, file);

        // Update to completed
        setDocuments((prev) =>
          prev.map((d) =>
            d.id === doc.id
              ? { ...d, status: 'completed', progress: 100 }
              : d
          )
        );
      } catch (error) {
        console.error('Upload failed:', error);
        setDocuments((prev) =>
          prev.map((d) =>
            d.id === doc.id
              ? { ...d, status: 'error', error: error instanceof Error ? error.message : 'FastAPI 연결 실패' }
              : d
          )
        );
      }
    }

    toast.success('모든 문서 인덱싱 완료');
  };

  const handleDelete = (id: string) => {
    setDocuments((prev) => prev.filter((d) => d.id !== id));
    vectorStore.deleteDocument(id);
    toast.success('문서가 삭제되었습니다');
  };

  const completedDocs = documents.filter((d) => d.status === 'completed').length;

  return (
    <div className="flex h-screen overflow-hidden bg-background">
      {/* Document Panel */}
      <div
        className={`border-r bg-card transition-all duration-300 ${
          showDocPanel ? 'w-80' : 'w-0'
        } overflow-hidden`}
      >
        <div className="flex flex-col h-full">
          <div className="p-4 border-b">
            <div className="flex items-center justify-between mb-4">
              <h2 className="font-semibold flex items-center gap-2">
                <FileText className="h-5 w-5 text-primary" />
                문서 라이브러리
              </h2>
              <Button
                variant="ghost"
                size="icon"
                onClick={() => setShowDocPanel(false)}
              >
                <PanelLeftClose className="h-5 w-5" />
              </Button>
            </div>
            <DocumentUpload onUpload={handleUpload} />
          </div>
          <div className="flex-1 overflow-y-auto p-4">
            <DocumentList documents={documents} onDelete={handleDelete} />
          </div>
        </div>
      </div>

      {/* Chat Panel */}
      <div className="flex-1 flex flex-col">
        <header className="border-b bg-card px-6 py-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              {!showDocPanel && (
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() => setShowDocPanel(true)}
                >
                  <PanelLeft className="h-5 w-5" />
                </Button>
              )}
              <div>
                <h1 className="text-lg font-semibold">RAG QA 챗봇</h1>
                <p className="text-xs text-muted-foreground">
                  {completedDocs}개 문서 인덱싱 완료
                </p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <ThemeToggle />
              <Button variant="ghost" size="icon">
                <Settings className="h-5 w-5" />
              </Button>
            </div>
          </div>
        </header>

        <main className="flex-1 overflow-hidden">
          <ChatInterface documentsIndexed={completedDocs} />
        </main>
      </div>
    </div>
  );
};

export default Index;
