import { useMemo, useState, type FC } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import type { ChatMessageProps, Source } from './types';

/** Map a [0..1] cosine score to a UX-friendly relevance bucket. */
function relevanceBucket(score: number | undefined): { label: string; cls: string; pct: number } {
  if (score === undefined || score === null || isNaN(score)) {
    return { label: 'Relevância desconhecida', cls: 'relevance-low', pct: 0 };
  }
  const pct = Math.max(0, Math.min(100, Math.round(score * 100)));
  if (score >= 0.6) return { label: 'Alta relevância', cls: 'relevance-high', pct };
  if (score >= 0.45) return { label: 'Relevância média', cls: 'relevance-medium', pct };
  return { label: 'Relevância baixa', cls: 'relevance-low', pct };
}

/** TJMG public PDF URL pattern, derived from filename. */
function buildPdfUrl(filename: string | undefined): string | null {
  if (!filename) return null;
  const base = filename.replace(/\.(doc|docx|pdf)$/i, '');
  return `http://www8.tjmg.jus.br/institucional/at/pdf/${base}.pdf`;
}

interface GroupedSource {
  doc_id: number | string;
  primary: Source;
  extraChunks: Source[];
  bestScore: number;
}

function groupSources(sources: Source[]): GroupedSource[] {
  const map = new Map<string | number, GroupedSource>();
  for (const s of sources) {
    const key = s.document_id ?? `${s.tipo}-${s.numero}-${s.ano}-${s.filename}`;
    const existing = map.get(key);
    const score = s.score ?? 0;
    if (!existing) {
      map.set(key, { doc_id: key, primary: s, extraChunks: [], bestScore: score });
    } else if (score > existing.bestScore) {
      existing.extraChunks.unshift(existing.primary);
      existing.primary = s;
      existing.bestScore = score;
    } else {
      existing.extraChunks.push(s);
    }
  }
  return Array.from(map.values()).sort((a, b) => b.bestScore - a.bestScore);
}

const ChatMessage: FC<ChatMessageProps> = ({ message }) => {
  const isUser = message.role === 'user';
  const roleClass = isUser ? 'user' : 'assistant';
  const grouped = useMemo(() => groupSources(message.sources ?? []), [message.sources]);
  // Sources expanded by default — confiança jurídica vem da fonte
  const [sourcesExpanded, setSourcesExpanded] = useState<boolean>(true);
  const [copied, setCopied] = useState<boolean>(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(message.content);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch {
      // navigator.clipboard may be blocked in some browsers/contexts; ignore silently
    }
  };

  return (
    <div className={`chat-message ${roleClass}`}>
      <div className={`chat-avatar ${roleClass}`} aria-hidden="true">
        {isUser ? (
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
            <circle cx="12" cy="7" r="4" />
          </svg>
        ) : (
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6" />
          </svg>
        )}
      </div>

      <div className="chat-content">
        <ReactMarkdown remarkPlugins={[remarkGfm]}>
          {message.content}
        </ReactMarkdown>

        {!isUser && message.content && (
          <div className="message-actions" role="group" aria-label="Ações da resposta">
            <button type="button" className="message-action-btn" onClick={handleCopy} aria-label="Copiar resposta">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
                <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
              </svg>
              {copied ? 'Copiado!' : 'Copiar'}
            </button>
          </div>
        )}

        {grouped.length > 0 && (
          <div className="sources-expander">
            <button
              type="button"
              className="sources-header"
              aria-expanded={sourcesExpanded}
              aria-controls={`sources-${message.role}-${grouped[0].doc_id}`}
              onClick={() => setSourcesExpanded((v) => !v)}
              style={{ width: '100%', border: 'none', textAlign: 'left', fontFamily: 'inherit' }}
            >
              <span>Fontes consultadas ({grouped.length})</span>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ transform: sourcesExpanded ? 'rotate(180deg)' : 'none', transition: 'transform 0.2s' }} aria-hidden="true">
                <polyline points="6 9 12 15 18 9" />
              </svg>
            </button>

            {sourcesExpanded && (
              <div id={`sources-${message.role}-${grouped[0].doc_id}`} className="sources-content">
                {grouped.map((g) => {
                  const src = g.primary;
                  let badgeClass = 'badge-unknown';
                  if (src.status === 'VIGENTE') badgeClass = 'badge-vigente';
                  if (src.status === 'REVOGADO') badgeClass = 'badge-revogado';

                  const rel = relevanceBucket(g.bestScore);
                  const pdfUrl = buildPdfUrl(src.filename);

                  return (
                    <div className="source-card" key={String(g.doc_id)}>
                      <div className="source-meta-row">
                        <span className="source-title" style={{ marginBottom: 0 }}>
                          {src.tipo || 'Ato'} {src.numero ?? ''}/{src.ano ?? ''}
                        </span>
                        {src.status && (
                          <span className={`badge ${badgeClass}`}>{src.status}</span>
                        )}
                        <span className={`relevance-badge ${rel.cls}`} title={`Score bruto: ${(g.bestScore ?? 0).toFixed(2)}`}>
                          <span className="relevance-bar" aria-hidden="true">
                            <span className="relevance-bar-fill" style={{ width: `${rel.pct}%` }} />
                          </span>
                          {rel.label}
                        </span>
                        {pdfUrl && (
                          <a
                            href={pdfUrl}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="source-link"
                            aria-label={`Abrir PDF original de ${src.tipo} ${src.numero}/${src.ano} (TJMG)`}
                          >
                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                              <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" />
                              <polyline points="15 3 21 3 21 9" />
                              <line x1="10" y1="14" x2="21" y2="3" />
                            </svg>
                            PDF
                          </a>
                        )}
                      </div>
                      {src.orgao && (
                        <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '0.3rem' }}>
                          {src.orgao}
                        </div>
                      )}
                      <div className="source-excerpt">
                        {src.chunk_text ? src.chunk_text.substring(0, 300) + '…' : ''}
                      </div>
                      {g.extraChunks.length > 0 && (
                        <div className="source-extra-chunks">
                          + {g.extraChunks.length} {g.extraChunks.length === 1 ? 'trecho adicional' : 'trechos adicionais'} deste documento
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default ChatMessage;
