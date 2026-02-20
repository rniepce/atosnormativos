import React from 'react';

const Sidebar = ({
    llmProvider,
    setLlmProvider,
    filterStatus,
    setFilterStatus,
    filterTipo,
    setFilterTipo,
    filterAno,
    setFilterAno,
    onClearChat
}) => {
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

            {/* Model Section */}
            <h3>🤖 Modelo</h3>
            <div className="form-group">
                <label>Provedor LLM</label>
                <select value={llmProvider} onChange={(e) => setLlmProvider(e.target.value)}>
                    <option value="anthropic">Claude 4.6 Sonnet</option>
                    <option value="amazonia">Amazônia IA</option>
                </select>
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
                    <option value="Resolução">Resolução</option>
                    <option value="Provimento">Provimento</option>
                    <option value="Recomendação">Recomendação</option>
                    <option value="Instrução Normativa">Instrução Normativa</option>
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
            <button className="btn-clear" onClick={onClearChat}>
                🗑️ Limpar Conversa
            </button>

            {/* Footer */}
            <div className="sidebar-footer">
                📂 Base: ~13.000 atos normativos<br />
                ⚖️ Tribunal de Justiça de Minas Gerais
            </div>
        </div>
    );
};

export default Sidebar;
