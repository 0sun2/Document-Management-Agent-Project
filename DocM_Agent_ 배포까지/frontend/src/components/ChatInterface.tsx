import { useState, useRef, useEffect } from 'react';
import { Send, Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { ChatMessage as ChatMessageType } from '@/types';
import { ChatMessage } from './ChatMessage';
import { llm } from '@/adapters/llm';
import { toast } from 'sonner';

interface ChatInterfaceProps {
  documentsIndexed: number;
}

export function ChatInterface({ documentsIndexed }: ChatInterfaceProps) {
  const [messages, setMessages] = useState<ChatMessageType[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    if (documentsIndexed === 0) {
      toast.error('먼저 문서를 업로드해주세요');
      return;
    }

    const userMessage: ChatMessageType = {
      id: Date.now().toString(),
      role: 'user',
      content: input.trim(),
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      // Backend handles everything: RAG search + LLM generation + web fallback
      const response = await llm.generateResponse(
        [...messages, userMessage],
        [] // Citations will come from backend
      );

      const assistantMessage: ChatMessageType = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: response.content,
        timestamp: new Date(),
        citations: response.citations || [],
        confidence: response.confidence,
        mode: response.mode,
        toolResults: response.toolResults,
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Chat error:', error);
      const errorMessage: ChatMessageType = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: 'FastAPI 백엔드 연결 실패. 서버가 실행 중인지 확인하세요. (http://127.0.0.1:8000)',
        timestamp: new Date(),
        confidence: 'low',
      };
      setMessages(prev => [...prev, errorMessage]);
      toast.error('FastAPI 백엔드 연결 실패');
    } finally {
      setIsLoading(false);
    }
  };

  const suggestedQuestions = [
    '이 문서의 핵심 내용을 요약해줘',
    '주요 주장 3개를 표로 정리해줘',
    '가장 중요한 인용구는 뭐야?',
  ];

  return (
    <div className="flex flex-col h-full">
      <div className="flex-1 overflow-y-auto px-4 py-6 space-y-6">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-center px-4">
            <h2 className="text-2xl font-semibold mb-2">
              내 문서에 바로 답하는 지능형 AI
            </h2>
            <p className="text-muted-foreground mb-8 max-w-lg">
              문서 우선 RAG, 필요할 땐 웹·도구로 보강. 출처까지 한 눈에.
            </p>
            {documentsIndexed > 0 && (
              <div className="space-y-2 w-full max-w-md">
                <p className="text-sm text-muted-foreground mb-3">예시 질문:</p>
                {suggestedQuestions.map((q, i) => (
                  <button
                    key={i}
                    onClick={() => setInput(q)}
                    className="w-full text-left px-4 py-3 rounded-lg border border-border hover:border-primary hover:bg-accent transition-colors text-sm"
                  >
                    {q}
                  </button>
                ))}
              </div>
            )}
          </div>
        ) : (
          messages.map(message => (
            <ChatMessage key={message.id} message={message} />
          ))
        )}
        {isLoading && (
          <div className="flex items-center gap-2 text-muted-foreground">
            <Loader2 className="h-4 w-4 animate-spin" />
            <span className="text-sm">생각하는 중...</span>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <div className="border-t bg-background p-4">
        <form onSubmit={handleSubmit} className="flex gap-2">
          <Textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="질문을 입력하세요..."
            className="min-h-[60px] resize-none"
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSubmit(e);
              }
            }}
          />
          <Button
            type="submit"
            size="icon"
            disabled={!input.trim() || isLoading}
            className="h-[60px] w-[60px] shrink-0"
          >
            <Send className="h-5 w-5" />
          </Button>
        </form>
      </div>
    </div>
  );
}
