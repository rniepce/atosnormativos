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
                <svg width="32" height="32" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M12 2L2 22H22L12 2Z" stroke="currentColor" strokeWidth="2" strokeLinejoin="round" fill="none" />
                </svg>
                <div style={{ marginLeft: "10px" }}>
                    <div className="sidebar-logo-text" style={{ fontSize: "16px", fontWeight: 700 }}>TJMG</div>
                    <div className="sidebar-logo-sub" style={{ fontSize: "10px", color: "gray" }}>ATOS NORMATIVOS</div>
                </div>
            </div>

            {/* Nav Items */}
            <div style={{ display: "flex", flexDirection: "column", gap: "20px", marginTop: "1rem" }}>
                
                {/* Filtros */}
                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", cursor: "pointer" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: "12px", color: "#374151" }}>
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <line x1="4" y1="6" x2="20" y2="6"></line>
                            <line x1="8" y1="12" x2="20" y2="12"></line>
                            <line x1="12" y1="18" x2="20" y2="18"></line>
                        </svg>
                        <span style={{ fontWeight: 500, fontSize: "15px" }}>Filtros</span>
                    </div>
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#9CA3AF" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="6 9 12 15 18 9"></polyline></svg>
                </div>

                <hr style={{ border: 0, borderTop: "1px solid #E5E7EB", margin: 0 }} />

                {/* Modelo */}
                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", cursor: "pointer" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: "12px", color: "#374151" }}>
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <circle cx="12" cy="12" r="3"></circle>
                            <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path>
                        </svg>
                        <div style={{ display: "flex", flexDirection: "column" }}>
                            <span style={{ fontWeight: 500, fontSize: "15px" }}>Modelo</span>
                            <span style={{ fontSize: "13px", color: "#6B7280" }}>{modelInfo.label}</span>
                        </div>
                    </div>
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#9CA3AF" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="9 18 15 12 9 6"></polyline></svg>
                </div>

                <hr style={{ border: 0, borderTop: "1px solid #E5E7EB", margin: 0 }} />

                {/* Base */}
                <div style={{ display: "flex", alignItems: "center", gap: "12px", color: "#374151" }}>
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <ellipse cx="12" cy="5" rx="9" ry="3"></ellipse>
                        <path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"></path>
                        <path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"></path>
                    </svg>
                    <span style={{ fontWeight: 500, fontSize: "15px" }}>Base: ~13.000 atos</span>
                </div>
            </div>

            <div className="sidebar-spacer" style={{ flexGrow: 1 }}></div>

            <div className="form-group upload-key-group" style={{ display: "none" }}>
                <input type="password" placeholder="Chave de Upload" value={uploadApiKey} onChange={(e) => setUploadApiKey(e.target.value)} />
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
                        <>
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                <path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"></path>
                                <polyline points="13 2 13 9 20 9"></polyline>
                            </svg>
                            Subir Ato Normativo
                        </>
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
