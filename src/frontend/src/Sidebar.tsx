import React, { useState, useEffect, type ChangeEvent } from 'react';
import type { SidebarProps, ModelInfo } from './types';

const Sidebar: React.FC<SidebarProps> = ({
    filterStatus,
    setFilterStatus,
    filterTipo,
    setFilterTipo,
    uploadApiKey,
    setUploadApiKey,
    onClearChat,
    onUploadFile,
    isUploading,
    isOpen,
    onClose,
}) => {
    const [modelInfo, setModelInfo] = useState<ModelInfo>({ label: 'Carregando...', provider: '' });
    const fileInputRef = React.useRef<HTMLInputElement>(null);

    const handleFileSelect = async (e: ChangeEvent<HTMLInputElement>): Promise<void> => {
        const file = e.target.files?.[0];
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
            .then((data: ModelInfo) => setModelInfo(data))
            .catch(() => setModelInfo({ label: 'Indisponível', provider: 'none' }));
    }, [backendUrl]);

    return (
        <div className={`sidebar ${isOpen ? 'open' : ''}`}>
            {/* Logo */}
            <div className="sidebar-logo">
                <svg width="32" height="32" viewBox="0 0 120 110" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M60 8 L110 100 L10 100 Z" stroke="currentColor" strokeWidth="10" strokeLinejoin="round" fill="none" />
                </svg>
                <div>
                    <div className="sidebar-logo-text">TJMG</div>
                    <div className="sidebar-logo-sub">ATOS NORMATIVOS</div>
                </div>
            </div>

            {/* Filters Section */}
            <div className="sidebar-section">
                <div className="section-header">
                    <span className="icon">≡</span> <h3>Filtros</h3>
                    <span className="chevron">⌄</span>
                </div>
                {/* Visual filter options can go here, for now we keep the functional selects visually hidden or restyled */}
                <div className="form-group">
                    <select value={filterStatus} onChange={(e) => setFilterStatus(e.target.value)}>
                        <option value="">Status: Todos</option>
                        <option value="VIGENTE">VIGENTE</option>
                        <option value="REVOGADO">REVOGADO</option>
                    </select>
                </div>
                <div className="form-group">
                    <select value={filterTipo} onChange={(e) => setFilterTipo(e.target.value)}>
                        <option value="">Tipo: Todos</option>
                        <option value="Portaria">Portaria</option>
                        <option value="Resolução">Resolução</option>
                        <option value="Provimento">Provimento</option>
                        <option value="Aviso">Aviso</option>
                    </select>
                </div>
            </div>

            {/* Model Section */}
            <div className="sidebar-section">
                <div className="section-header">
                    <span className="icon">⚙️</span>
                    <div>
                        <h3>Modelo</h3>
                        <div className="sub-text">{modelInfo.label}</div>
                    </div>
                    <span className="chevron">&gt;</span>
                </div>
            </div>

            {/* Base Stats Section */}
            <div className="sidebar-section">
                <div className="section-header">
                    <span className="icon">🗄️</span>
                    <h3>Base: ~13.000 atos</h3>
                </div>
            </div>

            <div className="sidebar-spacer" style={{ flexGrow: 1 }}></div>

            {/* Upload Key Section - Keep functional but unobtrusive */}
            <div className="form-group upload-key-group">
                <input
                    type="password"
                    placeholder="Chave de Upload (Opcional)"
                    value={uploadApiKey}
                    onChange={(e) => setUploadApiKey(e.target.value)}
                />
            </div>

            {/* Action Buttons */}
            <div className="sidebar-actions">
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
            </div>
        </div>
    );
};

export default Sidebar;
