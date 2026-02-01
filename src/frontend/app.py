import streamlit as st
import requests
import os
import json

# Backend URL
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8080")

st.set_page_config(page_title="TJMG Normativos RAG", layout="wide")

st.title("⚖️ Busca de Atos Normativos TJMG")
st.markdown("Sistema de busca inteligente com RAG (Retrieval-Augmented Generation)")

# Sidebar for filters
with st.sidebar:
    st.header("Filtros")
    filter_status = st.selectbox("Status", ["", "VIGENTE", "REVOGADO"], index=0)
    filter_tipo = st.selectbox("Tipo de Ato", ["", "Portaria", "Resolução", "Provimento", "Recomendação", "Instrução Normativa"], index=0)
    filter_ano = st.text_input("Ano (Ex: 2023)")

# Main search bar
query = st.text_input("Digite sua dúvida (Ex: Como funciona férias prêmio?)")

if st.button("Pesquisar", type="primary"):
    if not query:
        st.warning("Por favor, digite uma pergunta.")
    else:
        with st.spinner("Pesquisando e analisando atos normativos..."):
            try:
                # Prepare payload
                payload = {"query": query}
                if filter_status: payload["filter_status"] = filter_status
                if filter_tipo: payload["filter_tipo"] = filter_tipo
                if filter_ano: payload["filter_ano"] = int(filter_ano) if filter_ano.isdigit() else None

                response = requests.post(f"{BACKEND_URL}/search", json=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "")
                    sources = data.get("sources", [])

                    # Display Answer
                    st.subheader("Resposta Gerada (IA)")
                    st.markdown(answer)
                    
                    st.divider()
                    
                    # Display Sources
                    st.subheader("Fontes Consultadas")
                    for src in sources:
                        with st.expander(f"{src['tipo']} {src['numero']}/{src['ano']} - {src['status']} (Relevância: {src['score']:.2f})"):
                             st.markdown(f"**Trecho:**")
                             st.markdown(f"> {src['chunk_text']}")
                else:
                    st.error(f"Erro na busca: {response.text}")

            except Exception as e:
                st.error(f"Erro ao conectar com o backend: {e}")
