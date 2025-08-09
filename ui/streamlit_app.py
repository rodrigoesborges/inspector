# ui/streamlit_app.py
import streamlit as st
import requests
import os
from pathlib import Path

API_URL = os.environ.get("API_URL", "http://backend:8997")

# Set page config with custom icon
icon_path = Path(__file__).parent / "assets" / "favicon.png"

st.set_page_config(
    page_icon=str(icon_path)
)

st.title("Inspector-RAG (IPEA) — POC")
st.markdown("Pergunte sobre séries do IPEA. Você pode informar um `SERCODIGO` (opcional) ou deixar que o sistema busque automaticamente.")

sercodigo = st.text_input("SERCODIGO (opcional)", value="")
question = st.text_area("Pergunta (em Português)", value="Como evoluiu o IPCA entre 2010 e 2020?")
top_k = st.slider("Número de trechos recuperados (k)", 1, 10, value=5)

if st.button("Enviar"):
    if not question.strip():
        st.error("Coloque uma pergunta.")
    else:
        payload = {
            "question": question,
            "sercodigo": sercodigo.strip() or None,
            "top_k": top_k,
            "use_model": "openai",
            "model_name": "gpt-5-mini"

#            "use_model": "ollama",
#            "model_name": "llama3.2"
        }
        with st.spinner("Consultando backend..."):
            resp = requests.post(f"{API_URL}/query", json=payload, timeout=120)
        if resp.status_code != 200:
            st.error(f"Erro: {resp.status_code} - {resp.text}")
        else:
            js = resp.json()
            st.subheader("Resposta do modelo")
            st.write(js.get("llm_text"))
            st.subheader("Trechos recuperados")
            for hit in js.get("retrieval_hits", []):
                st.write(hit)
            st.subheader("Observações")
            st.write(js.get("indexed_note"))
