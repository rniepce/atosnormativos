import streamlit as st
import requests
import os

# Backend URL
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8080")

st.set_page_config(page_title="TJMG Normativos RAG", layout="wide", page_icon="⚖️")

# Custom CSS for chatbot style
st.markdown("""
<style>
.chat-message {
    padding: 1rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    display: flex;
    flex-direction: column;
}
.chat-message.user {
    background-color: #e3f2fd;
}
.chat-message.assistant {
    background-color: #f5f5f5;
}
.chat-message .content {
    margin-top: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/8a/Bras%C3%A3o_de_Minas_Gerais.svg/150px-Bras%C3%A3o_de_Minas_Gerais.svg.png", width=100)
    st.title("⚖️ Atos Normativos TJMG")
    st.markdown("---")
    
    # Tabs for different functionalities
    tab_selection = st.radio("Navegação", ["💬 Chat", "📤 Upload de PDFs"], label_visibility="collapsed")
    
    st.markdown("---")
    st.header("Filtros de Busca")
    filter_status = st.selectbox("Status", ["", "VIGENTE", "REVOGADO"], index=0)
    filter_tipo = st.selectbox("Tipo de Ato", ["", "Portaria", "Resolução", "Provimento", "Recomendação", "Instrução Normativa"], index=0)
    filter_ano = st.text_input("Ano (Ex: 2023)")
    
    if st.button("🗑️ Limpar Chat"):
        st.session_state.messages = []
        st.rerun()

# Main Content Area
if tab_selection == "📤 Upload de PDFs":
    st.header("📤 Upload de Atos Normativos")
    st.markdown("Faça upload de arquivos PDF para incluí-los na base de conhecimento do sistema RAG.")
    
    uploaded_files = st.file_uploader(
        "Arraste ou selecione arquivos PDF",
        type=["pdf"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        if st.button("🚀 Processar Arquivos", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, file in enumerate(uploaded_files):
                status_text.text(f"Processando: {file.name}...")
                
                try:
                    files = {"file": (file.name, file.getvalue(), "application/pdf")}
                    response = requests.post(f"{BACKEND_URL}/upload", files=files, timeout=300)
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.success(f"✅ **{file.name}** processado com sucesso!")
                        with st.expander(f"Detalhes: {file.name}"):
                            st.json(result)
                    else:
                        st.error(f"❌ Erro ao processar {file.name}: {response.text}")
                
                except Exception as e:
                    st.error(f"❌ Erro ao conectar com o backend: {e}")
                
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            status_text.text("Processamento concluído!")

else:  # Chat Tab
    st.header("💬 Consulte os Atos Normativos")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("📚 Fontes Consultadas"):
                    for src in message["sources"]:
                        st.markdown(f"**{src.get('tipo', '')} {src.get('numero', '')}/{src.get('ano', '')}** - {src.get('status', '')}")
                        st.markdown(f"> {src.get('chunk_text', '')[:300]}...")
                        st.markdown("---")
    
    # Chat input
    if prompt := st.chat_input("Digite sua dúvida sobre atos normativos..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Pesquisando e analisando..."):
                try:
                    # Prepare payload
                    payload = {"query": prompt}
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
                            with st.expander("📚 Fontes Consultadas"):
                                for src in sources:
                                    st.markdown(f"**{src.get('tipo', '')} {src.get('numero', '')}/{src.get('ano', '')}** - {src.get('status', '')} (Relevância: {src.get('score', 0):.2f})")
                                    st.markdown(f"> {src.get('chunk_text', '')[:300]}...")
                                    st.markdown("---")
                        
                        # Add assistant message to history
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources
                        })
                    else:
                        error_msg = f"Erro na busca: {response.text}"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
                
                except Exception as e:
                    error_msg = f"Erro ao conectar com o backend: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
