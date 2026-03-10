import React, { useState, useEffect } from 'react';

const Sidebar = ({
    filterStatus,
    setFilterStatus,
    filterTipo,
    setFilterTipo,
    filterAno,
    setFilterAno,
    googleApiKey,
    setGoogleApiKey,
    onClearChat,
    onUploadFile,
    isUploading,
    isOpen,
    onClose,
}) => {
    const [modelInfo, setModelInfo] = useState({ label: 'Carregando...', provider: '' });
    const [showKey, setShowKey] = useState(false);
    const fileInputRef = React.useRef(null);

    const handleFileSelect = async (e) => {
        const file = e.target.files[0];
        if (file) {
            await onUploadFile(file);
            e.target.value = '';
        }
    };

    const handleClearChat = () => {
        onClearChat();
        onClose();
    };

    const backendUrl = import.meta.env.PROD ? window.location.origin : (import.meta.env.VITE_BACKEND_URL || "http://localhost:8080");

    useEffect(() => {
        fetch(`${backendUrl}/api/model-info`)
            .then(res => res.json())
            .then(data => setModelInfo(data))
            .catch(() => setModelInfo({ label: 'Indisponível', provider: 'none' }));
    }, [backendUrl]);

    const providerBadge = modelInfo.provider === 'ollama'
        ? { text: 'LOCAL', color: '#00B894' }
        : modelInfo.provider === 'google'
            ? { text: 'GOOGLE AI', color: '#4285F4' }
            : modelInfo.provider === 'azure'
                ? { text: 'CLOUD', color: '#0984E3' }
                : { text: '—', color: '#636E72' };

    const hasKey = googleApiKey && googleApiKey.trim().length > 0;

    return (
        <div className={`sidebar ${isOpen ? 'open' : ''}`}>
            {/* Logo */}
            <div className="sidebar-logo">
                <svg width="60" height="55" viewBox="0 0 120 110" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M60 8 L110 100 L10 100 Z" stroke="white" strokeWidth="10" strokeLinejoin="round" fill="none" />
                </svg>
                <div>
                    <div className="sidebar-logo-text">TJMG</div>
                    <div className="sidebar-logo-sub">Atos Normativos</div>
                </div>
            </div>

            <hr />

            {/* Model Selector */}
            <h3>🤖 Modelo</h3>
            <div className="form-group">
                <label>LLM</label>
                <div style={{ color: 'rgba(255,255,255,0.85)', fontSize: '0.85rem', padding: '0.5rem 0', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    {modelInfo.label}
                    <span style={{
                        background: hasKey || modelInfo.provider !== 'google' ? providerBadge.color : '#636E72',
                        color: '#fff',
                        fontSize: '0.6rem',
                        padding: '2px 8px',
                        borderRadius: '10px',
                        fontWeight: 600,
                        letterSpacing: '0.05em',
                    }}>
                        {hasKey || modelInfo.provider !== 'google' ? providerBadge.text : 'SEM CHAVE'}
                    </span>
                </div>
            </div>

            {modelInfo.provider === 'google' && (
                <div className="form-group">
                    <label>🔑 Google API Key</label>
                    <div style={{ position: 'relative' }}>
                        <input
                            type={showKey ? 'text' : 'password'}
                            placeholder="Cole sua chave aqui..."
                            value={googleApiKey}
                            onChange={(e) => setGoogleApiKey(e.target.value)}
                            style={{ paddingRight: '2.5rem' }}
                        />
                        <button
                            type="button"
                            onClick={() => setShowKey(!showKey)}
                            style={{
                                position: 'absolute',
                                right: '6px',
                                top: '50%',
                                transform: 'translateY(-50%)',
                                background: 'none',
                                border: 'none',
                                color: 'rgba(255,255,255,0.5)',
                                cursor: 'pointer',
                                fontSize: '0.8rem',
                                padding: '4px',
                            }}
                        >
                            {showKey ? '🙈' : '👁️'}
                        </button>
                    </div>
                    <div style={{ fontSize: '0.65rem', color: 'rgba(255,255,255,0.4)', marginTop: '0.2rem' }}>
                        Grátis em <a href="https://aistudio.google.com/apikey" target="_blank" rel="noreferrer" style={{ color: '#4285F4' }}>aistudio.google.com</a>
                    </div>
                </div>
            )}

            <hr />

            {/* Filters Section */}
            <h3>🔍 Filtros</h3>
            <div className="form-group">
                <label>Status do Ato</label>
                <select value={filterStatus} onChange={(e) => setFilterStatus(e.target.value)}>
                    <option value="">Todos</option>
                    <option value="VIGENTE">VIGENTE</option>
                    <option value="REVOGADO">REVOGADO</option>
                </select>
            </div>

            <div className="form-group">
                <label>Tipo de Ato</label>
                <select value={filterTipo} onChange={(e) => setFilterTipo(e.target.value)}>
                    <option value="">Todos</option>
                    <option value="Portaria">Portaria</option>
                    <option value="Portaria Conjunta">Portaria Conjunta</option>
                    <option value="Resolução">Resolução</option>
                    <option value="Provimento Conjunto">Provimento</option>
                    <option value="Aviso">Aviso</option>
                    <option value="Ordem de Serviço">Ordem de Serviço</option>
                    <option value="Emenda Regimental">Emenda Regimental</option>
                </select>
            </div>

            <div className="form-group">
                <label>Ano</label>
                <input
                    type="text"
                    placeholder="Ex: 2023"
                    value={filterAno}
                    onChange={(e) => setFilterAno(e.target.value)}
                />
            </div>

            <hr />

            {/* Action Buttons */}
            <button className="btn-primary" onClick={handleClearChat}>
                ✨ Novo Chat
            </button>

            <button
                className="btn-upload"
                onClick={() => fileInputRef.current?.click()}
                disabled={isUploading}
            >
                {isUploading ? (
                    <>
                        <div className="spinner small" /> Processando...
                    </>
                ) : (
                    '📄 Subir Ato Normativo'
                )}
            </button>
            <input
                ref={fileInputRef}
                type="file"
                accept=".pdf,.doc,.docx"
                style={{ display: 'none' }}
                onChange={handleFileSelect}
            />

            {/* Footer */}
            <div className="sidebar-footer">
                📂 Base: ~13.000 atos normativos<br />
                ⚖️ Tribunal de Justiça de Minas Gerais
            </div>
        </div>
    );
};

export default Sidebar;
