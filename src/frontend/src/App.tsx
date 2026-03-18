import { useState, useRef, useEffect, type FormEvent, type KeyboardEvent, type ChangeEvent } from 'react';
import Sidebar from './Sidebar';
import ChatMessage from './ChatMessage';
import type { Message, SearchPayload, UploadResponse } from './types';

function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [prompt, setPrompt] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);

  // Sidebar states
  const [filterStatus, setFilterStatus] = useState('');
  const [filterTipo, setFilterTipo] = useState('');
  const [filterAno, setFilterAno] = useState('');
  const [selectedModel, setSelectedModel] = useState('GPT 4.1 mini');
  
  const [uploadApiKey, setUploadApiKey] = useState<string>(() => {
    try { return localStorage.getItem('upload_api_key') || ''; } catch { return ''; }
  });

  const chatEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom of chat
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    try { localStorage.setItem('upload_api_key', uploadApiKey); } catch { /* noop */ }
  }, [uploadApiKey]);

  // In production, the backend and frontend are hosted on the same origin
  const backendUrl = import.meta.env.PROD ? window.location.origin : (import.meta.env.VITE_BACKEND_URL || "http://localhost:8080");

  const clearChat = () => {
    setMessages([]);
  };

  const handleUploadFile = async (file: File): Promise<void> => {
    setIsUploading(true);

    // Show upload message
    setMessages(prev => [
      ...prev,
      { role: 'user', content: `📄 Subindo ato normativo: **${file.name}**` }
    ]);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const headers: Record<string, string> = {};
      if (uploadApiKey.trim()) {
        headers['X-API-Key'] = uploadApiKey.trim();
      }

      const response = await fetch(`${backendUrl}/upload`, {
        method: 'POST',
        headers,
        body: formData,
      });

      if (response.ok) {
        const data: UploadResponse = await response.json();
        const meta = data.metadata || {};
        const answer = `✅ **Ato normativo processado com sucesso!**\n\n` +
          `- **Arquivo:** ${data.filename}\n` +
          `- **Tipo:** ${meta.tipo || 'N/A'}\n` +
          `- **Número:** ${meta.numero || 'N/A'}/${meta.ano || 'N/A'}\n` +
          `- **Órgão:** ${meta.orgao || 'N/A'}\n` +
          `- **Status:** ${meta.status || 'N/A'}\n` +
          `- **Assunto:** ${meta.assunto_resumo || 'N/A'}\n` +
          `- **Chunks criados:** ${data.chunks_created}\n\n` +
          `O documento já está disponível para buscas.`;

        setMessages(prev => [
          ...prev,
          { role: 'assistant', content: answer }
        ]);
      } else {
        const errorText = await response.text();
        setMessages(prev => [
          ...prev,
          { role: 'assistant', content: `❌ Erro no upload: ${errorText}` }
        ]);
      }
    } catch (error) {
      const msg = error instanceof Error ? error.message : String(error);
      setMessages(prev => [
        ...prev,
        { role: 'assistant', content: `❌ Erro ao conectar: ${msg}` }
      ]);
    } finally {
      setIsUploading(false);
    }
  };

  const handleSubmit = async (e: FormEvent | KeyboardEvent): Promise<void> => {
    e.preventDefault();
    if (!prompt.trim() || isLoading) return;

    const userMessage: Message = { role: 'user', content: prompt };
    setMessages(prev => [...prev, userMessage]);
    setPrompt('');
    setIsLoading(true);

    try {
      const payload: SearchPayload = {
        query: userMessage.content,
      };

      if (filterStatus) payload.filter_status = filterStatus;
      if (filterTipo) payload.filter_tipo = filterTipo;
      if (filterAno && !isNaN(Number(filterAno))) payload.filter_ano = parseInt(filterAno, 10);
      if (selectedModel) payload.selected_model = selectedModel;

      const response = await fetch(`${backendUrl}/search`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });

      if (response.ok) {
        const data = await response.json();
        const answer: string = data.answer || "Não foi possível gerar uma resposta.";
        const sources = data.sources || [];

        setMessages(prev => [
          ...prev,
          { role: 'assistant', content: answer, sources }
        ]);
      } else {
        const errorText = await response.text();
        setMessages(prev => [
          ...prev,
          { role: 'assistant', content: `Erro na busca: ${errorText}` }
        ]);
      }
    } catch (error) {
      const msg = error instanceof Error ? error.message : String(error);
      setMessages(prev => [
        ...prev,
        { role: 'assistant', content: `Erro ao conectar com o backend: ${msg}` }
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <>
      {/* Hamburger button — visible only on mobile via CSS */}
      <button
        className={`hamburger-btn ${sidebarOpen ? 'active' : ''}`}
        onClick={() => setSidebarOpen(!sidebarOpen)}
        aria-label="Menu"
      >
        <span />
        <span />
        <span />
      </button>

      {/* Overlay behind sidebar on mobile */}
      <div
        className={`sidebar-overlay ${sidebarOpen ? 'visible' : ''}`}
        onClick={() => setSidebarOpen(false)}
      />

      <Sidebar
        filterStatus={filterStatus}
        setFilterStatus={setFilterStatus}
        filterTipo={filterTipo}
        setFilterTipo={setFilterTipo}
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
      />

      <main className="main-content">
        <div className="block-container">
          <div className="header-card">
            <div className="header-left">
              <h1 className="header-title">Consulta Inteligente</h1>
              <p className="header-subtitle">
                Pesquise portarias, resoluções, provimentos e demais atos normativos do TJMG utilizando inteligência artificial.
              </p>
            </div>
            <div className="header-right">
              {/* Removed User profile and Options */}
            </div>
          </div>

          <div className="chat-container">
            {messages.map((msg, idx) => (
              <ChatMessage key={idx} message={msg} />
            ))}

            {isLoading && (
              <div className="chat-message assistant">
                <div className="chat-avatar assistant">🤖</div>
                <div className="chat-content">
                  <div className="spinner-container">
                    <div className="spinner"></div>
                    Pesquisando e analisando…
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
              <textarea
                className="chat-input"
                placeholder="Faça sua pergunta sobre atos normativos…"
                value={prompt}
                onChange={(e: ChangeEvent<HTMLTextAreaElement>) => setPrompt(e.target.value)}
                onKeyDown={(e: KeyboardEvent<HTMLTextAreaElement>) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleSubmit(e);
                  }
                }}
                rows={1}
              />
              <div className="chat-input-actions">
                <button type="button" className="icon-btn" title="Anexar">🔗</button>
                <button type="button" className="icon-btn" title="Bot">🤖</button>
                <button
                  type="submit"
                  className="chat-submit-btn"
                  disabled={!prompt.trim() || isLoading}
                >
                  ➔
                </button>
              </div>
            </form>
          </div>
        </div>
      </main>
    </>
  );
}


export default App;
