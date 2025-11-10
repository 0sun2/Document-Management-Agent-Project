import { ChatMessage, Citation, ToolResult, WebSearchResultItem } from '@/types';

/**
 * LLM Adapter - FastAPI Backend (vLLM)
 * 
 * Connects to existing RAG pipeline:
 * - vLLM backend (Qwen/Qwen3-4B-Instruct-2507)
 * - Supabase pgvector for RAG search
 * - Serper API for web search fallback
 * 
 * Backend automatically:
 * 1. Searches documents via vector similarity
 * 2. Falls back to web search if similarity < threshold
 * 3. Generates answer with LLM
 */

const FASTAPI_URL = import.meta.env.VITE_FASTAPI_URL || 'http://127.0.0.1:8000';

interface AskApiMatch {
  doc_id: string;
  filename?: string;
  chunk_index?: number;
  similarity?: number;
  chunk_text?: string;
}

interface AskApiCitation {
  title?: string;
  url?: string;
  snippet?: string;
}

interface AskApiResponse {
  answer?: string;
  matches?: AskApiMatch[];
  citations?: AskApiCitation[];
  used_web_fallback?: boolean;
  source?: 'db' | 'web' | 'hybrid';
  top_similarity?: number;
}

export interface LLMAdapter {
  generateResponse(
    messages: ChatMessage[],
    citations: Citation[],
    systemPrompt?: string
  ): Promise<{
    content: string;
    confidence: 'low' | 'medium' | 'high';
    mode: 'docs-only' | 'web-augmented' | 'tool-fallback';
    citations?: Citation[];
    toolResults?: ToolResult[];
  }>;
}

class FastAPILLM implements LLMAdapter {
  async generateResponse(
    messages: ChatMessage[],
    citations: Citation[],
    systemPrompt?: string
  ): Promise<{
    content: string;
    confidence: 'low' | 'medium' | 'high';
    mode: 'docs-only' | 'web-augmented' | 'tool-fallback';
    citations?: Citation[];
    toolResults?: ToolResult[];
  }> {
    // Get the last user message
    const lastUserMessage = messages.filter(m => m.role === 'user').pop();
    if (!lastUserMessage) {
      throw new Error('No user message found');
    }

    const response = await fetch(`${FASTAPI_URL}/ask`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        question: lastUserMessage.content,
        top_k: 8,
      }),
    });

    if (!response.ok) {
      throw new Error('Failed to get response from backend');
    }

    const data: AskApiResponse = await response.json();

    // Parse backend response
    const usedWebFallback = data.used_web_fallback || false;
    const source = data.source || 'db';
    const topSimilarity = data.top_similarity || 0;

    // Map backend matches to Citations
    const backendCitations: Citation[] = (data.matches || []).map((match, idx) => ({
      id: `cite-${idx}`,
      documentId: match.doc_id,
      documentTitle: match.filename || 'Document',
      chunkIndex: match.chunk_index ?? 0,
      page: match.chunk_index ? Math.floor(match.chunk_index / 3) + 1 : 1,
      snippet: match.chunk_text || '',
      score: match.similarity || 0,
      highlight: match.chunk_text || '',
    }));

    // Map web citations
    const webCitations: Citation[] = (data.citations || []).map((cite, idx) => ({
      id: `web-${idx}`,
      documentId: 'web-search',
      documentTitle: cite.title || 'Web Result',
      chunkIndex: idx,
      page: 1,
      snippet: cite.snippet || cite.url || '',
      score: 0.9,
      highlight: cite.url || '',
    }));

    const allCitations = [...backendCitations, ...webCitations];

    // Tool results for web search
    const webSearchResults: WebSearchResultItem[] = (data.citations || []).map((cite) => ({
      title: cite.title || 'Web Result',
      snippet: cite.snippet || cite.url || '',
      url: cite.url || '',
    }));

    const toolResults: ToolResult[] = usedWebFallback ? [{
      toolName: 'web_search',
      query: lastUserMessage.content,
      result: { results: webSearchResults },
      timestamp: new Date(),
      type: 'web_search',
    }] : [];

    // Determine confidence based on similarity and source
    let confidence: 'low' | 'medium' | 'high';
    if (source === 'db' && topSimilarity > 0.85) {
      confidence = 'high';
    } else if (source === 'db' && topSimilarity > 0.7) {
      confidence = 'medium';
    } else {
      confidence = 'low';
    }

    // Determine mode
    let mode: 'docs-only' | 'web-augmented' | 'tool-fallback';
    if (usedWebFallback) {
      mode = 'web-augmented';
    } else if (source === 'db') {
      mode = 'docs-only';
    } else {
      mode = 'tool-fallback';
    }

    return {
      content: data.answer || 'No answer generated',
      confidence,
      mode,
      citations: allCitations,
      toolResults,
    };
  }
}

export const llm: LLMAdapter = new FastAPILLM();
