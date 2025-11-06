import { User, Bot, ChevronDown, ChevronUp, ExternalLink } from 'lucide-react';
import { useState } from 'react';
import { ChatMessage as ChatMessageType } from '@/types';
import { Badge } from '@/components/ui/badge';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';

interface ChatMessageProps {
  message: ChatMessageType;
}

export function ChatMessage({ message }: ChatMessageProps) {
  const [showEvidence, setShowEvidence] = useState(false);

  const isUser = message.role === 'user';

  return (
    <div className={`flex gap-3 ${isUser ? 'justify-end' : 'justify-start'}`}>
      {!isUser && (
        <div className="shrink-0 rounded-full bg-primary/10 p-2">
          <Bot className="h-5 w-5 text-primary" />
        </div>
      )}

      <div className={`flex flex-col gap-2 max-w-[80%] ${isUser ? 'items-end' : 'items-start'}`}>
        <Card className={`p-4 ${isUser ? 'bg-primary text-primary-foreground' : 'bg-card'}`}>
          <p className="text-sm whitespace-pre-wrap">{message.content}</p>
        </Card>

        {!isUser && (
          <div className="flex flex-wrap gap-2 items-center">
            {message.confidence && (
              <Badge
                variant="outline"
                className={
                  message.confidence === 'high'
                    ? 'bg-success/10 text-success border-success'
                    : message.confidence === 'medium'
                    ? 'bg-warning/10 text-warning border-warning'
                    : 'bg-destructive/10 text-destructive border-destructive'
                }
              >
                신뢰도: {message.confidence === 'high' ? '높음' : message.confidence === 'medium' ? '중간' : '낮음'}
              </Badge>
            )}

            {message.mode && (
              <Badge variant="secondary" className="text-xs">
                {message.mode === 'docs-only'
                  ? '📄 문서 기반'
                  : message.mode === 'web-augmented'
                  ? '🌐 웹 보강'
                  : '🔧 도구 사용'}
              </Badge>
            )}

            {message.citations && message.citations.length > 0 && (
              <Badge variant="outline" className="text-xs">
                {message.citations.length}개 출처
              </Badge>
            )}

            {(message.citations || message.toolResults) && (
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setShowEvidence(!showEvidence)}
                className="h-7 text-xs"
              >
                {showEvidence ? (
                  <>
                    <ChevronUp className="h-3 w-3 mr-1" />
                    근거 숨기기
                  </>
                ) : (
                  <>
                    <ChevronDown className="h-3 w-3 mr-1" />
                    근거 보기
                  </>
                )}
              </Button>
            )}
          </div>
        )}

        {showEvidence && (
          <Card className="w-full p-4 space-y-3">
            {message.citations && message.citations.length > 0 && (
              <div>
                <h4 className="font-semibold text-sm mb-2">📚 문서 인용</h4>
                <div className="space-y-2">
                  {message.citations.map((citation, i) => (
                    <div key={citation.id} className="text-sm border-l-2 border-primary pl-3 py-1">
                      <div className="flex items-start justify-between gap-2 mb-1">
                        <Badge variant="outline" className="text-xs shrink-0">
                          #{i + 1}
                        </Badge>
                        <p className="text-xs text-muted-foreground flex-1">
                          {citation.documentTitle}
                          {citation.page && ` · p.${citation.page}`}
                          <span className="ml-2 text-xs text-primary">
                            신뢰도: {(citation.score * 100).toFixed(0)}%
                          </span>
                        </p>
                      </div>
                      <p className="text-xs">{citation.snippet}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {message.toolResults && message.toolResults.length > 0 && (
              <div>
                <h4 className="font-semibold text-sm mb-2">🔧 외부 도구 결과</h4>
                <div className="space-y-2">
                  {message.toolResults.map((tool, i) => (
                    <Card key={i} className="p-3">
                      <div className="flex items-center gap-2 mb-2">
                        <Badge variant="secondary" className="text-xs">
                          {tool.toolName}
                        </Badge>
                        <span className="text-xs text-muted-foreground">
                          {tool.query}
                        </span>
                      </div>
                      {tool.type === 'web_search' && tool.result.results && (
                        <div className="space-y-2">
                          {tool.result.results.map((r: any, idx: number) => (
                            <div key={idx} className="text-xs">
                              <a
                                href={r.url}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="font-medium text-primary hover:underline inline-flex items-center gap-1"
                              >
                                {r.title}
                                <ExternalLink className="h-3 w-3" />
                              </a>
                              <p className="text-muted-foreground mt-1">{r.snippet}</p>
                            </div>
                          ))}
                        </div>
                      )}
                    </Card>
                  ))}
                </div>
              </div>
            )}
          </Card>
        )}
      </div>

      {isUser && (
        <div className="shrink-0 rounded-full bg-primary/10 p-2">
          <User className="h-5 w-5 text-primary" />
        </div>
      )}
    </div>
  );
}
