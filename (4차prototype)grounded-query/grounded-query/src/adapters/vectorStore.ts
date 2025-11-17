import { Citation } from '@/types';

/**
 * Vector Store Adapter - FastAPI Backend
 * 
 * Connects to existing vLLM + FastAPI backend
 * - POST /upload: document upload & embedding
 * - POST /ask: RAG search with vector similarity
 * 
 * Backend uses:
 * - sentence_transformers (multilingual-e5-large-instruct)
 * - Supabase pgvector for vector search
 * - Serper API for web search fallback
 */

const FASTAPI_URL = import.meta.env.VITE_FASTAPI_URL || 'http://127.0.0.1:8000';

export interface BackendDocumentSummary {
  id: string;
  title: string;
  chunks: number;
  sizeBytes: number;
  uploadedAt: string;
}

interface DocumentApiSummary {
  doc_id: string;
  filename: string;
  chunks: number;
  size_bytes: number;
  uploaded_at: string;
}

interface DocumentApiResponse {
  documents?: DocumentApiSummary[];
}

export interface VectorStoreAdapter {
  indexDocument(documentId: string, chunks: string[], filename: string, file: File): Promise<BackendDocumentSummary>;
  listDocuments(searchQuery?: string): Promise<BackendDocumentSummary[]>;
  search(query: string, topK: number, threshold: number): Promise<Citation[]>;
  deleteDocument(documentId: string): Promise<void>;
}

class FastAPIVectorStore implements VectorStoreAdapter {
  private docMetadata = new Map<string, { filename: string; docId: string }>();

  async indexDocument(documentId: string, chunks: string[], filename: string, file: File): Promise<BackendDocumentSummary> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('document_id', documentId);
    formData.append('original_filename', filename);

    const uploadUrl = `${FASTAPI_URL}/upload`;
    console.log('Uploading to:', uploadUrl);

    try {
      const response = await fetch(uploadUrl, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text();
        let errorData;
        try {
          errorData = JSON.parse(errorText);
        } catch {
          errorData = { error: errorText || 'Upload failed' };
        }
        console.error('Upload failed:', response.status, errorData);
        throw new Error(errorData.detail || errorData.error || `Upload failed: ${response.status}`);
      }

      const result = await response.json();

      // Store mapping between our frontend documentId and backend doc_id
      this.docMetadata.set(documentId, {
        filename: result.filename || filename,
        docId: result.doc_id,
      });

      return {
        id: result.doc_id,
        title: result.filename || filename,
        chunks: result.chunks || chunks.length || 0,
        sizeBytes: result.size_bytes ?? file.size,
        uploadedAt: result.uploaded_at ?? new Date().toISOString(),
      };
    } catch (error) {
      console.error('Upload error:', error);
      if (error instanceof TypeError && error.message.includes('fetch')) {
        throw new Error(`백엔드 서버에 연결할 수 없습니다. ${FASTAPI_URL} 서버가 실행 중인지 확인하세요.`);
      }
      throw error;
    }
  }

  async listDocuments(searchQuery?: string): Promise<BackendDocumentSummary[]> {
    try {
      const url = new URL(`${FASTAPI_URL}/documents`);
      if (searchQuery) {
        url.searchParams.set('q', searchQuery);
      }
      const response = await fetch(url);
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(errorText || '문서 목록을 불러오지 못했습니다');
      }
      const data: DocumentApiResponse = await response.json();
      return (data.documents || []).map((doc) => ({
        id: doc.doc_id,
        title: doc.filename,
        chunks: doc.chunks,
        sizeBytes: doc.size_bytes,
        uploadedAt: doc.uploaded_at,
      }));
    } catch (error) {
      console.error('List documents error:', error);
      throw error;
    }
  }

  async search(query: string, topK: number, threshold: number): Promise<Citation[]> {
    // This method is now handled by LLM adapter's generateResponse
    // but kept for interface compatibility
    return [];
  }

  async deleteDocument(documentId: string): Promise<void> {
    this.docMetadata.delete(documentId);
    try {
      await fetch(`${FASTAPI_URL}/documents/${documentId}`, {
        method: 'DELETE',
      });
    } catch (error) {
      console.warn('Failed to delete document from backend', error);
    }
  }

  getDocMetadata(documentId: string) {
    return this.docMetadata.get(documentId);
  }

  getAllDocs() {
    return Array.from(this.docMetadata.entries());
  }
}

export const vectorStore: FastAPIVectorStore = new FastAPIVectorStore();
