import { useEffect, useRef, useState, type ChangeEvent, type FC } from 'react';
import type { SidebarProps } from './types';
import logoUrl from '../assets/tjmg_logo.png';

const DEFAULT_TIPOS = [
  'Portaria',
  'Portaria Conjunta',
  'Resolução',
  'Provimento',
  'Provimento Conjunto',
  'Aviso',
  'Aviso Conjunto',
  'Instrução',
  'Instrução de Serviço',
  'Ordem de Serviço',
  'Regimento Interno',
  'Emenda Regimental',
  'Deliberação',
  'Enunciado',
  'Orientação Administrativa',
];

const Sidebar: FC<SidebarProps> = ({
  filterStatus,
  setFilterStatus,
  filterTipo,
  setFilterTipo,
  filterOrgao,
  setFilterOrgao,
  filterAno,
  setFilterAno,
  selectedModel,
  setSelectedModel,
  onClearChat,
  onUploadFile,
  isUploading,
  isOpen,
  onClose,
  facets,
}) => {
  // Filters expanded by default on desktop, collapsed on mobile
  const isDesktop = typeof window !== 'undefined' ? window.innerWidth > 768 : true;
  const [isFiltrosOpen, setIsFiltrosOpen] = useState<boolean>(isDesktop);
  const [isModeloOpen, setIsModeloOpen] = useState<boolean>(false);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = async (e: ChangeEvent<HTMLInputElement>): Promise<void> => {
    const file = e.target.files?.[0];
    if (file) {
      await onUploadFile(file);
      e.target.value = '';
    }
  };

  const handleClearChat = () => {
    if (window.confirm('Iniciar nova conversa? O histórico atual será apagado.')) {
      onClearChat();
      onClose();
    }
  };

  const tipos = (facets?.tipos && facets.tipos.length > 0) ? facets.tipos : DEFAULT_TIPOS;
  const orgaos = facets?.orgaos ?? [];
  const totalDocs = facets?.total ?? 12986;

  // Year range from facets (fallback 1961-current year)
  const currentYear = new Date().getFullYear();
  const minYear = facets?.anos.length ? Math.min(...facets.anos) : 1961;
  const maxYear = facets?.anos.length ? Math.max(...facets.anos) : currentYear;

  // Persist accordion state across remounts of the component (mobile open/close)
  useEffect(() => {
    setIsFiltrosOpen((prev) => prev || (typeof window !== 'undefined' && window.innerWidth > 768));
  }, []);

  return (
    <aside className={`sidebar ${isOpen ? 'open' : ''}`} aria-label="Barra lateral de filtros e ações">
      <div className="sidebar-logo">
        <img src={logoUrl} alt="" className="sidebar-logo-img" aria-hidden="true" />
        <div>
          <div className="sidebar-logo-text" style={{ fontSize: '1rem', fontWeight: 700 }}>TJMG</div>
          <div className="sidebar-logo-sub">ATOS NORMATIVOS</div>
        </div>
      </div>

      <nav style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem', marginTop: '0.5rem' }} aria-label="Configurações de busca">
        {/* ── Filtros ──────────────────────────────── */}
        <div className="sidebar-section">
          <button
            type="button"
            className="section-toggle"
            aria-expanded={isFiltrosOpen}
            aria-controls="filtros-body"
            onClick={() => setIsFiltrosOpen((v) => !v)}
          >
            <span className="section-toggle-label">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <line x1="4" y1="6" x2="20" y2="6" />
                <line x1="8" y1="12" x2="20" y2="12" />
                <line x1="12" y1="18" x2="20" y2="18" />
              </svg>
              <span className="section-toggle-title">Filtros</span>
            </span>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="section-chevron" aria-hidden="true">
              <polyline points="6 9 12 15 18 9" />
            </svg>
          </button>

          {isFiltrosOpen && (
            <div id="filtros-body" className="section-body">
              <div className="form-group" style={{ margin: 0 }}>
                <label htmlFor="filter-tipo" className="field-label">Tipo de Ato</label>
                <select id="filter-tipo" value={filterTipo} onChange={(e) => setFilterTipo(e.target.value)}>
                  <option value="">Todos</option>
                  {tipos.map((t) => (
                    <option key={t} value={t}>{t}</option>
                  ))}
                </select>
              </div>

              <div className="form-group" style={{ margin: 0 }}>
                <label htmlFor="filter-orgao" className="field-label">Órgão Emissor</label>
                <select id="filter-orgao" value={filterOrgao} onChange={(e) => setFilterOrgao(e.target.value)}>
                  <option value="">Todos</option>
                  {orgaos.map((o) => (
                    <option key={o} value={o}>{o}</option>
                  ))}
                </select>
                {orgaos.length === 0 && (
                  <small style={{ color: 'var(--text-muted)', fontSize: '0.7rem' }}>Carregando órgãos…</small>
                )}
              </div>

              <div className="form-group" style={{ margin: 0 }}>
                <label htmlFor="filter-ano" className="field-label">Ano</label>
                <input
                  id="filter-ano"
                  type="number"
                  inputMode="numeric"
                  min={minYear}
                  max={maxYear}
                  step={1}
                  placeholder={`${minYear} – ${maxYear}`}
                  value={filterAno}
                  onChange={(e) => setFilterAno(e.target.value)}
                />
              </div>

              <div className="form-group" style={{ margin: 0 }}>
                <label htmlFor="filter-status" className="field-label">Status</label>
                <select id="filter-status" value={filterStatus} onChange={(e) => setFilterStatus(e.target.value)}>
                  <option value="">Todos</option>
                  <option value="VIGENTE">Vigente</option>
                  <option value="REVOGADO">Revogado</option>
                </select>
              </div>
            </div>
          )}
        </div>

        <hr className="section-divider" />

        {/* ── Modelo ──────────────────────────────── */}
        <div className="sidebar-section">
          <button
            type="button"
            className="section-toggle"
            aria-expanded={isModeloOpen}
            aria-controls="modelo-body"
            onClick={() => setIsModeloOpen((v) => !v)}
          >
            <span className="section-toggle-label">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <circle cx="12" cy="12" r="3" />
                <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
              </svg>
              <span className="section-toggle-title">
                Modelo
                <span className="section-toggle-sub">{selectedModel}</span>
              </span>
            </span>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="section-chevron" aria-hidden="true">
              <polyline points="6 9 12 15 18 9" />
            </svg>
          </button>

          {isModeloOpen && (
            <div id="modelo-body" className="section-body">
              <div className="radio-group" role="radiogroup" aria-label="Selecionar modelo de IA">
                <label className="radio-option">
                  <input
                    type="radio"
                    name="modelSelection"
                    value="GPT 5.4 mini"
                    checked={selectedModel === 'GPT 5.4 mini'}
                    onChange={(e) => setSelectedModel(e.target.value)}
                  />
                  <span>
                    GPT 5.4 mini
                    <span className="radio-option-meta">Rápido, ideal para perguntas diretas</span>
                  </span>
                </label>
                <label className="radio-option">
                  <input
                    type="radio"
                    name="modelSelection"
                    value="GPT 5.2"
                    checked={selectedModel === 'GPT 5.2'}
                    onChange={(e) => setSelectedModel(e.target.value)}
                  />
                  <span>
                    GPT 5.2
                    <span className="radio-option-meta">Mais profundo, melhor para análises</span>
                  </span>
                </label>
              </div>
            </div>
          )}
        </div>

        <hr className="section-divider" />

        {/* ── Base info ───────────────────────────── */}
        <div className="sidebar-info">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
            <ellipse cx="12" cy="5" rx="9" ry="3" />
            <path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3" />
            <path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5" />
          </svg>
          <span>Base: {totalDocs.toLocaleString('pt-BR')} atos</span>
        </div>
      </nav>

      <div className="sidebar-spacer" style={{ flexGrow: 1 }} />

      <div className="sidebar-actions">
        <button type="button" className="btn-primary" onClick={handleClearChat}>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
            <line x1="12" y1="5" x2="12" y2="19" />
            <line x1="5" y1="12" x2="19" y2="12" />
          </svg>
          Nova consulta
        </button>

        <button
          type="button"
          className="btn-upload"
          onClick={() => fileInputRef.current?.click()}
          disabled={isUploading}
        >
          {isUploading ? (
            <>
              <span className="spinner small" /> Processando…
            </>
          ) : (
            <>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z" />
                <polyline points="13 2 13 9 20 9" />
              </svg>
              Subir ato normativo
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
    </aside>
  );
};

export default Sidebar;
