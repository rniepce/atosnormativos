import streamlit as st
import requests
import os

# Backend URL - ensure scheme is present
_backend_url = os.getenv("BACKEND_URL", "http://localhost:8080")
if _backend_url and not _backend_url.startswith(("http://", "https://")):
    _backend_url = f"https://{_backend_url}"
BACKEND_URL = _backend_url

st.set_page_config(
    page_title="TJMG — Consulta de Atos Normativos",
    layout="wide",
    page_icon="⚖️",
    initial_sidebar_state="expanded",
)


# --- Inline SVG: white outline triangle (TJMG style, transparent BG) ---
logo_html = '''<svg width="60" height="55" viewBox="0 0 120 110" fill="none" xmlns="http://www.w3.org/2000/svg">
  <path d="M60 8 L110 100 L10 100 Z" stroke="white" stroke-width="10" stroke-linejoin="round" fill="none"/>
</svg>'''


# ═══════════════════════════════════════════════════════════════
#  CUSTOM CSS — Modern, Clean, TJMG-branded
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Google Font ────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Root variables ─────────────────────────────── */
:root {
    --tjmg-red: #C8102E;
    --tjmg-red-light: #E94560;
    --tjmg-red-dark: #9B0C22;
    --tjmg-dark: #1A1A2E;
    --tjmg-dark-2: #16213E;
    --tjmg-dark-3: #0F3460;
    --surface: #F8F9FA;
    --surface-2: #FFFFFF;
    --text-primary: #2D3436;
    --text-secondary: #636E72;
    --text-muted: #B2BEC3;
    --border: #E0E0E0;
    --shadow-sm: 0 1px 3px rgba(0,0,0,0.06);
    --shadow-md: 0 4px 12px rgba(0,0,0,0.08);
    --shadow-lg: 0 8px 30px rgba(0,0,0,0.12);
    --radius: 12px;
    --radius-sm: 8px;
    --radius-lg: 16px;
    --transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
}

/* ── Global ─────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

.stApp {
    background: linear-gradient(135deg, #F5F7FA 0%, #F0F2F5 50%, #EBF0F5 100%);
}

/* ── Hide Streamlit Branding ────────────────────── */
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}
.stDeployButton {display: none;}

/* ── Sidebar ────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--tjmg-dark) 0%, var(--tjmg-dark-2) 100%) !important;
    border-right: none !important;
    box-shadow: 4px 0 20px rgba(0,0,0,0.15);
}

[data-testid="stSidebar"] * {
    color: #E8E8E8 !important;
}

[data-testid="stSidebar"] .stMarkdown p {
    color: #B0B8C4 !important;
    font-size: 0.85rem;
}

[data-testid="stSidebar"] h1 {
    color: #FFFFFF !important;
    font-size: 1.3rem !important;
    font-weight: 600 !important;
    letter-spacing: -0.02em;
}

[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: #FFFFFF !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 1.2rem !important;
    margin-bottom: 0.5rem !important;
}

/* Sidebar dividers */
[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.08) !important;
    margin: 1rem 0 !important;
}

/* Sidebar selectbox & inputs */
[data-testid="stSidebar"] .stSelectbox > div > div,
[data-testid="stSidebar"] .stTextInput > div > div > input {
    background-color: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: var(--radius-sm) !important;
    color: #FFFFFF !important;
    transition: var(--transition);
}

[data-testid="stSidebar"] .stSelectbox > div > div:hover,
[data-testid="stSidebar"] .stTextInput > div > div > input:hover {
    border-color: var(--tjmg-red-light) !important;
    background-color: rgba(255,255,255,0.1) !important;
}

[data-testid="stSidebar"] .stSelectbox > div > div:focus-within,
[data-testid="stSidebar"] .stTextInput > div > div > input:focus {
    border-color: var(--tjmg-red) !important;
    box-shadow: 0 0 0 2px rgba(200,16,46,0.25) !important;
}

/* Sidebar labels */
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stTextInput label {
    color: rgba(255,255,255,0.65) !important;
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Sidebar button */
[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    border: 1px solid rgba(233,69,96,0.5) !important;
    color: var(--tjmg-red-light) !important;
    border-radius: var(--radius-sm) !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    padding: 0.5rem 1.2rem !important;
    transition: var(--transition);
    width: 100%;
}

[data-testid="stSidebar"] .stButton > button:hover {
    background: var(--tjmg-red) !important;
    border-color: var(--tjmg-red) !important;
    color: #FFFFFF !important;
    box-shadow: 0 4px 15px rgba(200,16,46,0.35) !important;
    transform: translateY(-1px);
}

/* ── Main Content ───────────────────────────────── */
.main .block-container {
    max-width: 900px !important;
    padding: 2rem 2rem 6rem 2rem !important;
}

/* ── Header Card ────────────────────────────────── */
.header-card {
    background: var(--surface-2);
    border-radius: var(--radius-lg);
    padding: 1.8rem 2rem;
    margin-bottom: 1.5rem;
    box-shadow: var(--shadow-sm);
    border: 1px solid var(--border);
    position: relative;
    overflow: hidden;
}

.header-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--tjmg-red), var(--tjmg-red-light), var(--tjmg-red));
    background-size: 200% 100%;
    animation: shimmer 3s ease-in-out infinite;
}

@keyframes shimmer {
    0%, 100% { background-position: -200% 0; }
    50% { background-position: 200% 0; }
}

.header-title {
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0 0 0.3rem 0;
    letter-spacing: -0.02em;
}

.header-subtitle {
    font-size: 0.9rem;
    color: var(--text-secondary);
    margin: 0;
    font-weight: 400;
}

/* ── Chat Messages ──────────────────────────────── */
[data-testid="stChatMessage"] {
    background: var(--surface-2) !important;
    border-radius: var(--radius) !important;
    border: 1px solid var(--border) !important;
    box-shadow: var(--shadow-sm) !important;
    padding: 1rem 1.2rem !important;
    margin-bottom: 0.8rem !important;
    animation: fadeInUp 0.3s ease-out;
}

@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(8px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* User message accent */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    border-left: 3px solid var(--tjmg-red) !important;
}

/* Assistant message accent */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    border-left: 3px solid var(--tjmg-dark-3) !important;
    background: linear-gradient(135deg, #FFFFFF 0%, #FAFBFC 100%) !important;
}

/* Chat input */
[data-testid="stChatInput"] {
    border-radius: var(--radius) !important;
}

[data-testid="stChatInput"] textarea {
    border-radius: var(--radius) !important;
    border: 2px solid var(--border) !important;
    background: var(--surface-2) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
    transition: var(--transition);
    padding: 0.8rem 1rem !important;
}

[data-testid="stChatInput"] textarea:focus {
    border-color: var(--tjmg-red) !important;
    box-shadow: 0 0 0 3px rgba(200,16,46,0.15) !important;
}

/* Chat submit button */
[data-testid="stChatInput"] button {
    background: var(--tjmg-red) !important;
    border-radius: var(--radius-sm) !important;
    transition: var(--transition);
}

[data-testid="stChatInput"] button:hover {
    background: var(--tjmg-red-dark) !important;
    transform: scale(1.05);
}

/* ── Expanders (Sources) ────────────────────────── */
.streamlit-expanderHeader {
    background: var(--surface) !important;
    border-radius: var(--radius-sm) !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    color: var(--text-secondary) !important;
    border: 1px solid var(--border) !important;
    transition: var(--transition);
}

.streamlit-expanderHeader:hover {
    background: #EDF2F7 !important;
    color: var(--text-primary) !important;
}

.streamlit-expanderContent {
    border: 1px solid var(--border) !important;
    border-top: none !important;
    border-radius: 0 0 var(--radius-sm) var(--radius-sm) !important;
    background: var(--surface-2) !important;
}

/* ── Status Badges ──────────────────────────────── */
.badge-vigente {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    background: linear-gradient(135deg, #00B894, #00CEC9);
    color: #FFFFFF;
    text-transform: uppercase;
}

.badge-revogado {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    background: linear-gradient(135deg, #636E72, #B2BEC3);
    color: #FFFFFF;
    text-transform: uppercase;
}

.badge-unknown {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    background: linear-gradient(135deg, #FDCB6E, #E17055);
    color: #FFFFFF;
    text-transform: uppercase;
}

/* ── Source Card ─────────────────────────────────── */
.source-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 0.8rem 1rem;
    margin-bottom: 0.6rem;
    transition: var(--transition);
}

.source-card:hover {
    border-color: var(--tjmg-red-light);
    box-shadow: var(--shadow-sm);
}

.source-title {
    font-weight: 600;
    font-size: 0.88rem;
    color: var(--text-primary);
    margin-bottom: 0.3rem;
}

.source-excerpt {
    font-size: 0.8rem;
    color: var(--text-secondary);
    line-height: 1.5;
    margin-top: 0.4rem;
    padding-left: 0.6rem;
    border-left: 2px solid var(--border);
}

/* ── Scrollbar ──────────────────────────────────── */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: transparent;
}
::-webkit-scrollbar-thumb {
    background: #CCC;
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: #AAA;
}

/* ── Sidebar Logo Section ───────────────────────── */
.sidebar-logo {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    padding: 0.5rem 0 0.8rem 0;
}

.sidebar-logo-text {
    font-size: 1.1rem;
    font-weight: 700;
    color: #FFFFFF !important;
    line-height: 1.2;
    letter-spacing: -0.01em;
}

.sidebar-logo-sub {
    font-size: 0.7rem;
    color: rgba(255,255,255,0.5) !important;
    font-weight: 400;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

/* ── Footer ─────────────────────────────────────── */
.sidebar-footer {
    font-size: 0.72rem;
    color: rgba(255,255,255,0.35) !important;
    line-height: 1.5;
    padding-top: 0.5rem;
}

/* ── Spinner ────────────────────────────────────── */
.stSpinner > div {
    border-top-color: var(--tjmg-red) !important;
}

/* ── Empty state ────────────────────────────────── */
.empty-state {
    text-align: center;
    padding: 3rem 2rem;
    color: var(--text-muted);
}

.empty-state-icon {
    font-size: 3rem;
    margin-bottom: 1rem;
    opacity: 0.5;
}

.empty-state-text {
    font-size: 1rem;
    font-weight: 400;
    color: var(--text-secondary);
    margin-bottom: 0.3rem;
}

.empty-state-hint {
    font-size: 0.82rem;
    color: var(--text-muted);
}

</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  SESSION STATE
# ═══════════════════════════════════════════════════════════════
if "messages" not in st.session_state:
    st.session_state.messages = []


# ═══════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    # Logo section
    st.markdown(f"""
    <div class="sidebar-logo">
        {logo_html}
        <div>
            <div class="sidebar-logo-text">TJMG</div>
            <div class="sidebar-logo-sub">Atos Normativos</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # LLM section
    st.markdown("### 🤖 Modelo")
    llm_options = {"Gemini": "gemini", "Amazônia IA": "amazonia"}
    llm_label = st.selectbox("Provedor LLM", list(llm_options.keys()), index=0, label_visibility="collapsed")
    llm_provider = llm_options[llm_label]

    st.markdown("---")

    # Filters section
    st.markdown("### 🔍 Filtros")
    filter_status = st.selectbox("Status do Ato", ["Todos", "VIGENTE", "REVOGADO"], index=0)
    if filter_status == "Todos":
        filter_status = ""

    filter_tipo = st.selectbox(
        "Tipo de Ato",
        ["Todos", "Portaria", "Resolução", "Provimento", "Recomendação", "Instrução Normativa"],
        index=0,
    )
    if filter_tipo == "Todos":
        filter_tipo = ""

    filter_ano = st.text_input("Ano", placeholder="Ex: 2023")

    st.markdown("---")

    # Clear chat button
    if st.button("🗑️  Limpar Conversa"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")

    # Footer
    st.markdown("""
    <div class="sidebar-footer">
        📂 Base: ~13.000 atos normativos<br/>
        ⚖️ Tribunal de Justiça de Minas Gerais
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  MAIN CONTENT — HEADER
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<div class="header-card">
    <div class="header-title">💬 Consulta Inteligente</div>
    <div class="header-subtitle">
        Pesquise portarias, resoluções, provimentos e demais atos normativos do TJMG
        utilizando inteligência artificial.
    </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  CHAT HISTORY
# ═══════════════════════════════════════════════════════════════
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander(f"📚 Fontes Consultadas ({len(message['sources'])})"):
                for src in message["sources"]:
                    status = src.get("status", "")
                    badge_class = "badge-vigente" if status == "VIGENTE" else ("badge-revogado" if status == "REVOGADO" else "badge-unknown")
                    badge_html = f'<span class="{badge_class}">{status}</span>' if status else ""

                    st.markdown(f"""
                    <div class="source-card">
                        <div class="source-title">
                            {src.get('tipo', '')} {src.get('numero', '')}/{src.get('ano', '')}
                            &nbsp;{badge_html}
                        </div>
                        <div class="source-excerpt">{src.get('chunk_text', '')[:300]}…</div>
                    </div>
                    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  CHAT INPUT
# ═══════════════════════════════════════════════════════════════
if prompt := st.chat_input("Faça sua pergunta sobre atos normativos…"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Pesquisando e analisando…"):
            try:
                # Prepare payload
                payload = {"query": prompt, "llm_provider": llm_provider}
                if filter_status:
                    payload["filter_status"] = filter_status
                if filter_tipo:
                    payload["filter_tipo"] = filter_tipo
                if filter_ano and filter_ano.isdigit():
                    payload["filter_ano"] = int(filter_ano)

                response = requests.post(f"{BACKEND_URL}/search", json=payload, timeout=120)

                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "Não foi possível gerar uma resposta.")
                    sources = data.get("sources", [])

                    st.markdown(answer)

                    if sources:
                        with st.expander(f"📚 Fontes Consultadas ({len(sources)})"):
                            for src in sources:
                                status_val = src.get("status", "")
                                badge_cls = "badge-vigente" if status_val == "VIGENTE" else ("badge-revogado" if status_val == "REVOGADO" else "badge-unknown")
                                badge_h = f'<span class="{badge_cls}">{status_val}</span>' if status_val else ""

                                st.markdown(f"""
                                <div class="source-card">
                                    <div class="source-title">
                                        {src.get('tipo', '')} {src.get('numero', '')}/{src.get('ano', '')}
                                        &nbsp;{badge_h}
                                        <span style="float:right;font-size:0.72rem;color:var(--text-muted);">
                                            Score: {src.get('score', 0):.2f}
                                        </span>
                                    </div>
                                    <div class="source-excerpt">{src.get('chunk_text', '')[:300]}…</div>
                                </div>
                                """, unsafe_allow_html=True)

                    # Save to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    })
                else:
                    error_msg = f"Erro na busca: {response.text}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

            except Exception as e:
                error_msg = f"Erro ao conectar com o backend: {e}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
