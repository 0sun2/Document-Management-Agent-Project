import { FileText, Trash2, Tag, CheckCircle2, Loader2, AlertCircle } from 'lucide-react';
import { Document } from '@/types';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';

interface DocumentListProps {
  documents: Document[];
  onDelete: (id: string) => void;
}

export function DocumentList({ documents, onDelete }: DocumentListProps) {
  if (documents.length === 0) {
    return (
      <div className="text-center py-12 text-muted-foreground">
        <FileText className="h-12 w-12 mx-auto mb-4 opacity-50" />
        <p>아직 업로드된 문서가 없습니다</p>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {documents.map((doc) => (
        <Card key={doc.id} className="p-4">
          <div className="flex items-start gap-3">
            <div className="rounded-lg bg-primary/10 p-2.5">
              <FileText className="h-5 w-5 text-primary" />
            </div>
            <div className="flex-1 min-w-0">
              <div className="flex items-start justify-between gap-2 mb-2">
                <h4 className="font-medium text-sm truncate">{doc.title}</h4>
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-8 w-8 shrink-0"
                  onClick={() => onDelete(doc.id)}
                >
                  <Trash2 className="h-4 w-4" />
                </Button>
              </div>

              <div className="flex items-center gap-2 mb-2 text-xs text-muted-foreground">
                <span className="uppercase">{doc.type}</span>
                <span>•</span>
                <span>{(doc.size / 1024).toFixed(1)} KB</span>
                {doc.chunks && (
                  <>
                    <span>•</span>
                    <span>{doc.chunks} chunks</span>
                  </>
                )}
              </div>

              {doc.tags.length > 0 && (
                <div className="flex flex-wrap gap-1 mb-2">
                  {doc.tags.map((tag, i) => (
                    <Badge key={i} variant="secondary" className="text-xs">
                      {tag}
                    </Badge>
                  ))}
                </div>
              )}

              <div className="flex items-center gap-2">
                {doc.status === 'pending' && (
                  <Badge variant="outline" className="text-xs">
                    <Loader2 className="h-3 w-3 mr-1 animate-spin" />
                    대기중
                  </Badge>
                )}
                {doc.status === 'indexing' && (
                  <>
                    <Badge variant="outline" className="text-xs">
                      <Loader2 className="h-3 w-3 mr-1 animate-spin" />
                      인덱싱 중
                    </Badge>
                    {doc.progress !== undefined && (
                      <div className="flex-1">
                        <Progress value={doc.progress} className="h-1.5" />
                      </div>
                    )}
                  </>
                )}
                {doc.status === 'completed' && (
                  <Badge className="text-xs bg-success">
                    <CheckCircle2 className="h-3 w-3 mr-1" />
                    완료
                  </Badge>
                )}
                {doc.status === 'error' && (
                  <Badge variant="destructive" className="text-xs">
                    <AlertCircle className="h-3 w-3 mr-1" />
                    오류
                  </Badge>
                )}
              </div>

              {doc.error && (
                <p className="text-xs text-destructive mt-2">{doc.error}</p>
              )}
            </div>
          </div>
        </Card>
      ))}
    </div>
  );
}
