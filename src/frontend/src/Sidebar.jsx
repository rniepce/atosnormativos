import React from 'react';

const Sidebar = ({
    filterStatus,
    setFilterStatus,
    filterTipo,
    setFilterTipo,
    filterAno,
    setFilterAno,
    selectedModel,
    setSelectedModel,
    useEnrichedPrompt,
    setUseEnrichedPrompt,
    onClearChat,
    onUploadFile,
    isUploading,
}) => {
    const fileInputRef = React.useRef(null);

    const handleFileSelect = async (e) => {
        const file = e.target.files[0];
        if (file) {
            await onUploadFile(file);
            e.target.value = ''; // Reset input
        }
    };

    return (
        <div className="sidebar">
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
                <select value={selectedModel} onChange={(e) => setSelectedModel(e.target.value)}>
                    <option value="gpt-4.1-mini">GPT-4.1-mini (rápido)</option>
                    <option value="gpt-5.2-chat">GPT-5.2 (avançado)</option>
                </select>
            </div>

            {/* Enriched Prompt Toggle */}
            <div className="form-group">
                <label className="toggle-label">
                    <span>Prompt Enriquecido</span>
                    <div
                        className={`toggle-switch ${useEnrichedPrompt ? 'active' : ''}`}
                        onClick={() => setUseEnrichedPrompt(!useEnrichedPrompt)}
                    >
                        <div className="toggle-thumb" />
                    </div>
                </label>
                <span className="toggle-hint">
                    {useEnrichedPrompt
                        ? '✅ Contexto TJMG ativo — expansão de query + persona institucional'
                        : '⚡ Modo direto — query literal sem expansão'}
                </span>
            </div>

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
            <button className="btn-primary" onClick={onClearChat}>
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
