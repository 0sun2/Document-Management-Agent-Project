export interface Document {
  id: string;
  title: string;
  type: 'pdf' | 'docx' | 'txt' | 'md' | 'csv' | 'pptx';
  size: number;
  uploadedAt: Date;
  status: 'pending' | 'indexing' | 'completed' | 'error';
  progress?: number;
  tags: string[];
  chunks?: number;
  tokens?: number;
  error?: string;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  citations?: Citation[];
  confidence?: 'low' | 'medium' | 'high';
  mode?: 'docs-only' | 'web-augmented' | 'tool-fallback';
  toolResults?: ToolResult[];
}

export interface Citation {
  id: string;
  documentId: string;
  documentTitle: string;
  chunkIndex: number;
  page?: number;
  snippet: string;
  score: number;
  highlight?: string;
}

export type ToolType = 'web_search' | 'calculator' | 'fetch_url' | 'custom';

export interface WebSearchResultItem {
  title: string;
  snippet: string;
  url: string;
}

export interface WebSearchToolPayload {
  results: WebSearchResultItem[];
}

export interface CalculatorToolPayload {
  value?: number;
  error?: string;
}

export interface FetchUrlToolPayload {
  content?: string;
  title?: string;
}

type CustomToolPayload = Record<string, unknown>;

interface BaseToolResult<TType extends ToolType, TResult> {
  toolName: string;
  query: string;
  result: TResult;
  timestamp: Date;
  type: TType;
}

export type ToolResult =
  | BaseToolResult<'web_search', WebSearchToolPayload>
  | BaseToolResult<'calculator', CalculatorToolPayload>
  | BaseToolResult<'fetch_url', FetchUrlToolPayload>
  | BaseToolResult<'custom', CustomToolPayload>;

export interface RAGSettings {
  topK: number;
  scoreThreshold: number;
  reranking: boolean;
  windowSize: number;
  chunkOverlap: number;
  docsFirst: boolean;
}

export interface LLMSettings {
  endpoint: string;
  model: string;
  tokenLimit: number;
  systemPrompt: string;
}

export interface ToolConfig {
  enabled: boolean;
  name: string;
  description: string;
  params?: Record<string, unknown>;
}
