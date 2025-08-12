# ui/streamlit_app.py
import streamlit as st
import requests
import os
import pandas as pd
import altair as alt
import requests
from pathlib import Path

API_URL = os.environ.get("API_URL", "http://backend:8997")


# --- FUNÇÃO COM CACHE DE 5 MINUTOS ---
@st.cache_data(ttl=300) # ttl=300 segundos => 5 minutos
def fetch_indexed_series_list():
    """
    Busca a lista de séries indexadas do backend.
    O resultado desta função será cacheado pelo Streamlit por 5 minutos.
    """
    try:
        response = requests.get(f"{API_URL}/indexed_series", timeout=60)
        if response.status_code == 200:
            return response.json()
        else:
            # Retorna um dicionário vazio em caso de erro para não quebrar a UI
            st.error(f"Não foi possível buscar a lista de séries indexadas (Erro {response.status_code}).")
            return {"series": [], "total": 0}
    except requests.RequestException:
        st.error("Falha de conexão com o backend.")
        return {"series": [], "total": 0}


# Set page config with custom icon
icon_path = Path(__file__).parent / "assets" / "favicon.png"

st.set_page_config(
    page_title="Inspector-RAG (IPEADATA)",
    page_icon=str(icon_path)
)

st.title("Inspector-RAG (IPEADATA) — PoC")
st.markdown("Pergunte sobre séries agregadas pelo IPEADATA. Você pode informar um `SERCODIGO` (opcional) ou deixar que o sistema busque automaticamente.")

# --- Inicialização do Estado da Sessão ---
if "series_candidates" not in st.session_state:
    st.session_state.series_candidates = None
if "final_answer" not in st.session_state:
    st.session_state.final_answer = None
if "selected_series_code" not in st.session_state:
    st.session_state.selected_series_code = None

# --- LÓGICA DE EXIBIÇÃO NA SIDEBAR ---
# Coloque este bloco de código na parte principal do seu script Streamlit

with st.sidebar:
    st.header("Séries Indexadas")

    # Chama a função cacheada para obter os dados
    indexed_data = fetch_indexed_series_list()
    total_series = indexed_data.get("total", 0)

    st.metric(label="Total de Séries Únicas no Índice", value=total_series)
    
    # Botão para forçar a limpeza do cache e a atualização imediata
    if st.button("Atualizar Lista Agora"):
        st.cache_data.clear()
        st.rerun()

    with st.expander("Ver todas as séries", expanded=False):
        if total_series > 0:
            # Converte a lista de dicionários para um DataFrame do Pandas para exibição
            df_series = pd.DataFrame(indexed_data.get("series", []))
            # Usa o dataframe do Streamlit para uma tabela interativa com busca
            st.dataframe(df_series, use_container_width=True)
        else:
            st.write("Nenhuma série encontrada ou o backend está indisponível.")

# --- Área de Input do Usuário ---
with st.container(border=True):
    question = st.text_area(
        "Pergunta",
        value="Como evoluiu o abate de frangos entre 2010 e 2020?",
        help="Seja específico sobre o período de tempo para obter melhores resultados."
    )
    # O `top_k` agora se refere ao número de séries candidatas a serem encontradas
    #top_k = st.slider("Número de séries candidatas para buscar (k)", 1, 10, value=3)
    top_k = 5

        # --- NOVO WIDGET DE UPLOAD AQUI ---
    uploaded_file = st.file_uploader(
        "Anexar um documento (PDF, DOCX, TXT)",
        type=['pdf', 'docx', 'txt', 'md']
    )

    if st.button("1. Buscar Séries Relevantes e Gerar Análise", type="primary"):
        if not question.strip():
            st.error("Por favor, insira uma pergunta.")
        else:
            # ETAPA 1: Chamar um novo endpoint para encontrar a série
            with st.spinner("Buscando séries correspondentes..."):
                try:
                    payload = {"question": question, "top_k": top_k}
                    # IMPORTANTE: Usaremos um novo endpoint `/find_series`
                    resp = requests.post(f"{API_URL}/find_series", json=payload, timeout=60)
                    if resp.status_code == 200:
                        st.session_state.series_candidates = resp.json().get("series", [])
                        st.session_state.final_answer = None # Limpa a resposta anterior
                    else:
                        st.error(f"Erro ao buscar séries: {resp.status_code} - {resp.text}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Erro de conexão com o backend: {e}")

# --- Área de Seleção da Série ---
if st.session_state.series_candidates is not None:
    if not st.session_state.series_candidates:
        st.warning("Nenhuma série correspondente foi encontrada para a sua pergunta.")
    else:
 
        st.session_state.selected_series_code = st.session_state.series_candidates[0]['sercodigo']

        #if st.button("3. Gerar Análise", type="primary"):
        # ETAPA 2: Chamar o endpoint de query com a série confirmada
        with st.spinner(f"Analisando a série {st.session_state.selected_series_code}...Isso pode levar um minuto."):
            try:
                payload = {
                    "question": question,
                    "sercodigo": st.session_state.selected_series_code,
                    "use_model": "openai",
                    "model_name": "gpt-4o-mini"
                }

                files_to_send = {}
                if uploaded_file is not None:
                # Prepara o arquivo para ser enviado na requisição multipart
                    files_to_send['attachment'] = (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)

                resp = requests.post(
                    f"{API_URL}/query", 
                    data=payload, 
                    files=files_to_send, 
                    timeout=180
                )                    


                if resp.status_code == 200:
                    st.session_state.final_answer = resp.json()
                else:
                    st.error(f"Erro ao gerar análise: {resp.status_code} - {resp.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"Erro de conexão com o backend: {e}")

# --- Área de Exibição dos Resultados ---
if st.session_state.final_answer:
    st.markdown("---")
    st.subheader("✅ Análise Concluída")
    
    answer = st.session_state.final_answer
    

    # 1. Pega a resposta de texto do LLM.
    llm_response_text = answer.get("llm_text", "Nenhuma resposta gerada pelo modelo.")

    # 2. Substitui os delimitadores customizados do seu LLM pelos delimitadores padrão de LaTeX.
    #    Troca "[ " por "$$" e " ]" por "$$".
    formatted_text = llm_response_text.replace("[ ", "$$").replace(" ]", "$$")
    
    # 3. Usa st.markdown para renderizar o texto. Ele vai interpretar o $$...$$
    #    e formatar a equação matemática corretamente.
    st.markdown(formatted_text, unsafe_allow_html=True)

    

    if answer.get("chart_data"):
        st.subheader("Visualização dos Dados")
        df_chart = pd.DataFrame(answer["chart_data"])
        df_chart['date'] = pd.to_datetime(df_chart['date'])
        
        chart = alt.Chart(df_chart).mark_line().encode(
            x=alt.X('date:T', title='Data'),
            y=alt.Y('value:Q', title='Valor'),
            tooltip=['date:T', 'value:Q']
        ).interactive()
        
        st.altair_chart(chart, use_container_width=True)

    with st.expander("Ver contexto enviado ao modelo"):
        st.text(answer.get("context_used", "Nenhum contexto foi gerado."))


    # --- CORREÇÃO AQUI: Adiciona um expander para o contexto ---
    st.subheader("Saiba mais")
    with st.expander("Inspecionar o contexto enviado ao modelo de linguagem (LLM)"):
        # O st.text é ideal para exibir blocos de texto pré-formatados
        context_text = answer.get("context_used", "Nenhum contexto foi retornado pela API.")
        st.text(context_text)
    # --- FIM DA CORREÇÃO ---