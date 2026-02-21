import { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import Sidebar from './Sidebar';

function App() {
  const [messages, setMessages] = useState([]);
  const [prompt, setPrompt] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  // Sidebar states
  const [filterStatus, setFilterStatus] = useState('');
  const [filterTipo, setFilterTipo] = useState('');
  const [filterAno, setFilterAno] = useState('');

  const chatEndRef = useRef(null);

  // Auto-scroll to bottom of chat
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // In production, the backend and frontend are hosted on the same origin
  const backendUrl = import.meta.env.PROD ? window.location.origin : (import.meta.env.VITE_BACKEND_URL || "http://localhost:8080");

  const clearChat = () => {
    setMessages([]);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!prompt.trim() || isLoading) return;

    const userMessage = { role: 'user', content: prompt };
    setMessages(prev => [...prev, userMessage]);
    setPrompt('');
    setIsLoading(true);

    try {
      const payload = {
        query: userMessage.content,
      };

      if (filterStatus) payload.filter_status = filterStatus;
      if (filterTipo) payload.filter_tipo = filterTipo;
      if (filterAno && !isNaN(filterAno)) payload.filter_ano = parseInt(filterAno, 10);

      const response = await fetch(`${backendUrl}/search`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });

      if (response.ok) {
        const data = await response.json();
        const answer = data.answer || "Não foi possível gerar uma resposta.";
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
      setMessages(prev => [
        ...prev,
        { role: 'assistant', content: `Erro ao conectar com o backend: ${error.message}` }
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div id="root">
      <Sidebar
        filterStatus={filterStatus}
        setFilterStatus={setFilterStatus}
        filterTipo={filterTipo}
        setFilterTipo={setFilterTipo}
        filterAno={filterAno}
        setFilterAno={setFilterAno}
        onClearChat={clearChat}
      />

      <main className="main-content">
        <div className="block-container">
          <div className="header-card">
            <h1 className="header-title">💬 Consulta Inteligente</h1>
            <p className="header-subtitle">
              Pesquise portarias, resoluções, provimentos e demais atos normativos do TJMG
              utilizando inteligência artificial.
            </p>
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
                onChange={(e) => setPrompt(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleSubmit(e);
                  }
                }}
                disabled={isLoading}
                rows={1}
              />
              <button
                type="submit"
                className="chat-submit-btn"
                disabled={!prompt.trim() || isLoading}
              >
                ↑
              </button>
            </form>
          </div>
        </div>
      </main>
    </div>
  );
}

const ChatMessage = ({ message }) => {
  const isUser = message.role === 'user';
  const roleClass = isUser ? 'user' : 'assistant';
  const [sourcesExpanded, setSourcesExpanded] = useState(false);

  return (
    <div className={`chat-message ${roleClass}`}>
      <div className={`chat-avatar ${roleClass}`}>
        {isUser ? '👤' : '🤖'}
      </div>

      <div className="chat-content">
        <ReactMarkdown remarkPlugins={[remarkGfm]}>
          {message.content}
        </ReactMarkdown>

        {message.sources && message.sources.length > 0 && (
          <div className="sources-expander">
            <div
              className="sources-header"
              onClick={() => setSourcesExpanded(!sourcesExpanded)}
            >
              <span>📚 Fontes Consultadas ({message.sources.length})</span>
              <span>{sourcesExpanded ? '▼' : '▶'}</span>
            </div>

            {sourcesExpanded && (
              <div className="sources-content">
                {message.sources.map((src, idx) => {
                  let badgeClass = 'badge-unknown';
                  if (src.status === 'VIGENTE') badgeClass = 'badge-vigente';
                  if (src.status === 'REVOGADO') badgeClass = 'badge-revogado';

                  return (
                    <div className="source-card" key={idx}>
                      <div className="source-title">
                        {src.tipo} {src.numero}/{src.ano}
                        {src.status && (
                          <span className={`badge ${badgeClass}`}>
                            {src.status}
                          </span>
                        )}
                        {src.score !== undefined && (
                          <span className="source-score">
                            Score: {src.score.toFixed(2)}
                          </span>
                        )}
                      </div>
                      <div className="source-excerpt">
                        {src.chunk_text ? src.chunk_text.substring(0, 300) + '…' : ''}
                      </div>
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

export default App;
