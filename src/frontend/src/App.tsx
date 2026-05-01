import { useEffect, useRef, useState, type ChangeEvent, type FormEvent, type KeyboardEvent } from 'react';
import Sidebar from './Sidebar';
import ChatMessage from './ChatMessage';
import type { FacetsResponse, Message, SearchPayload, UploadResponse } from './types';

const EXAMPLE_QUERIES: { tag: string; text: string }[] = [
  { tag: 'Eleições', text: 'Como é feita a eleição da Presidência do TJMG?' },
  { tag: 'Recente', text: 'Resoluções de 2024 sobre teletrabalho' },
  { tag: 'Histórico', text: 'Atos mais antigos sobre férias forenses' },
  { tag: 'Vigência', text: 'Quais portarias da Corregedoria foram revogadas em 2023?' },
  { tag: 'Estrutura', text: 'O que diz o Regimento Interno sobre o Órgão Especial?' },
  { tag: 'Comparativo', text: 'Diferença entre Provimento e Provimento Conjunto' },
];

function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [prompt, setPrompt] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);

  // Filters
  const [filterStatus, setFilterStatus] = useState('');
  const [filterTipo, setFilterTipo] = useState('');
  const [filterOrgao, setFilterOrgao] = useState('');
  const [filterAno, setFilterAno] = useState('');
  const [selectedModel, setSelectedModel] = useState('GPT 5.4 mini');

  const [facets, setFacets] = useState<FacetsResponse | null>(null);

  const [uploadApiKey, setUploadApiKey] = useState<string>(() => {
    try { return localStorage.getItem('upload_api_key') || ''; } catch { return ''; }
  });

  const chatEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  useEffect(() => {
    try { localStorage.setItem('upload_api_key', uploadApiKey); } catch { /* noop */ }
  }, [uploadApiKey]);

  const backendUrl = import.meta.env.PROD
    ? window.location.origin
    : (import.meta.env.VITE_BACKEND_URL || 'http://localhost:8080');

  // Load facets once on mount
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const r = await fetch(`${backendUrl}/api/facets`);
        if (!r.ok) return;
        const data = (await r.json()) as FacetsResponse;
        if (!cancelled) setFacets(data);
      } catch {
        // facets are optional — sidebar falls back to defaults
      }
    })();
    return () => { cancelled = true; };
  }, [backendUrl]);

  // Auto-grow textarea height
  const handlePromptChange = (e: ChangeEvent<HTMLTextAreaElement>) => {
    setPrompt(e.target.value);
    const ta = e.target;
    ta.style.height = 'auto';
    ta.style.height = `${Math.min(ta.scrollHeight, 200)}px`;
  };

  const clearChat = () => setMessages([]);

  const handleUploadFile = async (file: File): Promise<void> => {
    setIsUploading(true);
    setMessages((prev) => [
      ...prev,
      { role: 'user', content: `📄 Subindo ato normativo: **${file.name}**` },
    ]);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const headers: Record<string, string> = {};
      if (uploadApiKey.trim()) headers['X-API-Key'] = uploadApiKey.trim();

      const response = await fetch(`${backendUrl}/upload`, {
        method: 'POST',
        headers,
        body: formData,
      });

      if (response.ok) {
        const data: UploadResponse = await response.json();
        const meta = data.metadata || {};
        const answer =
          `✅ **Ato normativo processado com sucesso!**\n\n` +
          `- **Arquivo:** ${data.filename}\n` +
          `- **Tipo:** ${meta.tipo || 'N/A'}\n` +
          `- **Número:** ${meta.numero || 'N/A'}/${meta.ano || 'N/A'}\n` +
          `- **Órgão:** ${meta.orgao || 'N/A'}\n` +
          `- **Status:** ${meta.status || 'N/A'}\n` +
          `- **Assunto:** ${meta.assunto_resumo || 'N/A'}\n` +
          `- **Chunks criados:** ${data.chunks_created}\n\n` +
          `O documento já está disponível para buscas.`;
        setMessages((prev) => [...prev, { role: 'assistant', content: answer }]);
      } else {
        const errorText = await response.text();
        setMessages((prev) => [...prev, { role: 'assistant', content: `❌ Erro no upload: ${errorText}` }]);
      }
    } catch (error) {
      const msg = error instanceof Error ? error.message : String(error);
      setMessages((prev) => [...prev, { role: 'assistant', content: `❌ Erro ao conectar: ${msg}` }]);
    } finally {
      setIsUploading(false);
    }
  };

  const submitQuery = async (queryText: string): Promise<void> => {
    if (!queryText.trim() || isLoading) return;

    const userMessage: Message = { role: 'user', content: queryText };
    setMessages((prev) => [...prev, userMessage]);
    setPrompt('');
    if (textareaRef.current) textareaRef.current.style.height = 'auto';
    setIsLoading(true);

    try {
      const payload: SearchPayload = { query: queryText };
      if (filterStatus) payload.filter_status = filterStatus;
      if (filterTipo) payload.filter_tipo = filterTipo;
      if (filterOrgao) payload.filter_orgao = filterOrgao;
      if (filterAno && !isNaN(Number(filterAno))) payload.filter_ano = parseInt(filterAno, 10);
      if (selectedModel) payload.selected_model = selectedModel;

      const response = await fetch(`${backendUrl}/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (response.ok) {
        const data = await response.json();
        const answer: string = data.answer || 'Não foi possível gerar uma resposta.';
        const sources = data.sources || [];
        setMessages((prev) => [...prev, { role: 'assistant', content: answer, sources }]);
      } else {
        const errorText = await response.text();
        setMessages((prev) => [...prev, { role: 'assistant', content: `Erro na busca: ${errorText}` }]);
      }
    } catch (error) {
      const msg = error instanceof Error ? error.message : String(error);
      setMessages((prev) => [...prev, { role: 'assistant', content: `Erro ao conectar com o backend: ${msg}` }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmit = (e: FormEvent | KeyboardEvent): void => {
    e.preventDefault();
    void submitQuery(prompt);
  };

  // Active filter chips for visibility (sidebar configs persist across "new chat")
  const activeFilters: { key: string; label: string; clear: () => void }[] = [];
  if (filterTipo) activeFilters.push({ key: 'tipo', label: `Tipo: ${filterTipo}`, clear: () => setFilterTipo('') });
  if (filterOrgao) activeFilters.push({ key: 'orgao', label: `Órgão: ${filterOrgao}`, clear: () => setFilterOrgao('') });
  if (filterAno) activeFilters.push({ key: 'ano', label: `Ano: ${filterAno}`, clear: () => setFilterAno('') });
  if (filterStatus) activeFilters.push({ key: 'status', label: `Status: ${filterStatus}`, clear: () => setFilterStatus('') });

  const isEmpty = messages.length === 0 && !isLoading;

  return (
    <>
      <a href="#chat-input" className="skip-link">Pular para o campo de busca</a>

      <button
        className={`hamburger-btn ${sidebarOpen ? 'active' : ''}`}
        onClick={() => setSidebarOpen(!sidebarOpen)}
        aria-label={sidebarOpen ? 'Fechar menu lateral' : 'Abrir menu lateral'}
        aria-expanded={sidebarOpen}
        aria-controls="app-sidebar"
      >
        <span /><span /><span />
      </button>

      <div
        className={`sidebar-overlay ${sidebarOpen ? 'visible' : ''}`}
        onClick={() => setSidebarOpen(false)}
        aria-hidden="true"
      />

      <div id="app-sidebar">
        <Sidebar
          filterStatus={filterStatus}
          setFilterStatus={setFilterStatus}
          filterTipo={filterTipo}
          setFilterTipo={setFilterTipo}
          filterOrgao={filterOrgao}
          setFilterOrgao={setFilterOrgao}
          filterAno={filterAno}
          setFilterAno={setFilterAno}
          selectedModel={selectedModel}
          setSelectedModel={setSelectedModel}
          uploadApiKey={uploadApiKey}
          setUploadApiKey={setUploadApiKey}
          onClearChat={clearChat}
          onUploadFile={handleUploadFile}
          isUploading={isUploading}
          isOpen={sidebarOpen}
          onClose={() => setSidebarOpen(false)}
          facets={facets}
        />
      </div>

      <main className="main-content" role="main">
        <div className="block-container">
          {!isEmpty && (
            <header className="header-card">
              <div className="header-left">
                <h1 className="header-title">Consulta Inteligente</h1>
                <p className="header-subtitle">
                  Pesquise atos normativos do TJMG com inteligência artificial.
                  {activeFilters.length > 0 && (
                    <span className="active-filters" role="list" aria-label="Filtros ativos">
                      {activeFilters.map((f) => (
                        <span className="filter-chip" key={f.key} role="listitem">
                          {f.label}
                          <button
                            type="button"
                            className="filter-chip-remove"
                            aria-label={`Remover filtro ${f.label}`}
                            onClick={f.clear}
                          >
                            ×
                          </button>
                        </span>
                      ))}
                    </span>
                  )}
                </p>
              </div>
            </header>
          )}

          <div
            className="chat-container"
            aria-live="polite"
            aria-busy={isLoading}
            aria-relevant="additions"
          >
            {isEmpty && (
              <section className="empty-state" aria-label="Sugestões de busca">
                <h1 className="empty-state-title">Consulta Inteligente</h1>
                <p className="empty-state-subtitle">
                  Pesquise atos normativos do TJMG (portarias, resoluções, provimentos e mais)
                  com inteligência artificial. Faça uma pergunta em linguagem natural ou comece
                  por um dos exemplos abaixo.
                </p>
                <div className="empty-state-stats">
                  <span><strong>{(facets?.total ?? 12986).toLocaleString('pt-BR')}</strong> atos indexados</span>
                  <span>De <strong>{facets?.anos.length ? Math.min(...facets.anos) : 1961}</strong> a <strong>{facets?.anos.length ? Math.max(...facets.anos) : new Date().getFullYear()}</strong></span>
                  <span><strong>Busca semântica + filtros</strong></span>
                </div>
                <div className="example-grid">
                  {EXAMPLE_QUERIES.map((q) => (
                    <button
                      key={q.text}
                      type="button"
                      className="example-card"
                      onClick={() => void submitQuery(q.text)}
                    >
                      <span className="example-card-tag">{q.tag}</span>
                      {q.text}
                    </button>
                  ))}
                </div>
                {activeFilters.length > 0 && (
                  <div className="active-filters" role="list" aria-label="Filtros ativos">
                    {activeFilters.map((f) => (
                      <span className="filter-chip" key={f.key} role="listitem">
                        {f.label}
                        <button
                          type="button"
                          className="filter-chip-remove"
                          aria-label={`Remover filtro ${f.label}`}
                          onClick={f.clear}
                        >
                          ×
                        </button>
                      </span>
                    ))}
                  </div>
                )}
              </section>
            )}

            {messages.map((msg, idx) => (
              <ChatMessage key={idx} message={msg} />
            ))}

            {isLoading && (
              <div className="chat-message assistant" role="status" aria-label="Buscando resposta">
                <div className="chat-avatar assistant" aria-hidden="true">
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6" />
                  </svg>
                </div>
                <div className="chat-content">
                  <div className="spinner-container">
                    <div className="spinner" aria-hidden="true"></div>
                    Buscando trechos relevantes e gerando resposta…
                  </div>
                </div>
              </div>
            )}
            <div ref={chatEndRef} />
          </div>
        </div>

        <div className="chat-input-container">
          <div className="chat-input-wrapper">
            <form onSubmit={handleSubmit}>
              <label htmlFor="chat-input" className="sr-only" style={{ position: 'absolute', width: 1, height: 1, padding: 0, margin: -1, overflow: 'hidden', clip: 'rect(0,0,0,0)', whiteSpace: 'nowrap', border: 0 }}>
                Faça sua pergunta sobre atos normativos
              </label>
              <textarea
                id="chat-input"
                ref={textareaRef}
                className="chat-input"
                placeholder="Faça sua pergunta sobre atos normativos…"
                value={prompt}
                onChange={handlePromptChange}
                onKeyDown={(e: KeyboardEvent<HTMLTextAreaElement>) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleSubmit(e);
                  }
                }}
                rows={1}
              />
              <div className="chat-input-actions">
                <button
                  type="submit"
                  className="chat-submit-btn"
                  disabled={!prompt.trim() || isLoading}
                  aria-label="Enviar pergunta"
                >
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                    <line x1="12" y1="19" x2="12" y2="5" />
                    <polyline points="5 12 12 5 19 12" />
                  </svg>
                </button>
              </div>
            </form>
          </div>
          <p className="ai-disclaimer">
            Respostas geradas por inteligência artificial. Sempre verifique a fonte original (PDF) antes de usar oficialmente.
          </p>
        </div>
      </main>
    </>
  );
}

export default App;
