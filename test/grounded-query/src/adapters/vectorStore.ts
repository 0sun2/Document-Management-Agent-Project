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

    const response = await fetch(`${FASTAPI_URL}/upload`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ error: 'Upload failed' }));
      throw new Error(error.error || 'Failed to upload document');
    }

    const result = await response.json();
    
    // Store mapping between our frontend documentId and backend doc_id
    this.docMetadata.set(documentId, {
      filename: result.filename || filename,
      docId: result.doc_id,
    });
  }

  async search(query: string, topK: number, threshold: number): Promise<Citation[]> {
    // This method is now handled by LLM adapter's generateResponse
    // but kept for interface compatibility
    return [];
  }

  async deleteDocument(documentId: string): Promise<void> {
    // Backend doesn't expose delete endpoint yet
    // Remove from local metadata for now
    this.docMetadata.delete(documentId);
  }

  getDocMetadata(documentId: string) {
    return this.docMetadata.get(documentId);
  }

  getAllDocs() {
    return Array.from(this.docMetadata.entries());
  }
}

export const vectorStore: FastAPIVectorStore = new FastAPIVectorStore();
