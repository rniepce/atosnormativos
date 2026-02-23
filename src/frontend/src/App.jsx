import { useState, useRef, useEffect } from 'react';
import Sidebar from './Sidebar';
import ChatMessage from './ChatMessage';

function App() {
  const [messages, setMessages] = useState([]);
  const [prompt, setPrompt] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  // Sidebar states
  const [filterStatus, setFilterStatus] = useState('');
  const [filterTipo, setFilterTipo] = useState('');
  const [filterAno, setFilterAno] = useState('');
  const [selectedModel, setSelectedModel] = useState('gpt-4.1-mini');

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
        model: selectedModel,
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
        selectedModel={selectedModel}
        setSelectedModel={setSelectedModel}
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


export default App;
