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

export interface VectorStoreAdapter {
  indexDocument(documentId: string, chunks: string[], filename: string, file: File): Promise<void>;
  search(query: string, topK: number, threshold: number): Promise<Citation[]>;
  deleteDocument(documentId: string): Promise<void>;
}

class FastAPIVectorStore implements VectorStoreAdapter {
  private docMetadata = new Map<string, { filename: string; docId: string }>();

  async indexDocument(documentId: string, chunks: string[], filename: string, file: File): Promise<void> {
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
    } catch (error) {
      console.error('Upload error:', error);
      if (error instanceof TypeError && error.message.includes('fetch')) {
        throw new Error(`백엔드 서버에 연결할 수 없습니다. ${FASTAPI_URL} 서버가 실행 중인지 확인하세요.`);
      }
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
