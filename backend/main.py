# backend/main.py
from fastapi import FastAPI, HTTPException, Form, File, UploadFile
from pydantic import BaseModel
from typing import Optional
import ipeadatapy as ip
import pandas as pd
import requests
import re
from contextlib import asynccontextmanager
from tools.ipeadata import search_metadata_by_keyword, get_series_values, get_metadata_by_sercodigo
from rag.retrieval import index_ipea_series, retrieve_similar, build_context_from_results
from llm.ollama_client import generate_answer
from rag.embedding import RedisVectorStore # Sua classe
import uvicorn
import os
from model.config_schema import (
    AppConfig, 
    REDIS_CLIENT,
    TIKA_SERVER_ENDPOINT
    )

# --- CACHE DE METADADOS NO BACKEND ---
# Usaremos um dicionário para cachear o DataFrame de metadados na memória.
# Isso evita o custo de chamar ip.metadata() em todas as requisições.
app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Carrega o cache na inicialização do servidor
    print("Carregando metadados do IPEA para o cache...")
    app_state["metadata_df"] = ip.list_series()
    print("✅ Cache de metadados carregado.")
    yield
     # Código que executa no desligamento (shutdown)
    print("Limpando cache...")
    app_state.clear()

app = FastAPI(title="IPEADATA-RAG-Redis-Backend-POC",lifespan=lifespan)
store = RedisVectorStore()

# --- Modelos Pydantic para validação ---
class FindRequest(BaseModel):
    question: str
    top_k: int = 3
    
class QueryRequest(BaseModel):
    question: str
    sercodigo: str   # optional direct series code
    use_model: Optional[str] = "openai"  # or 'openai'
    model_name: Optional[str] = "gpt-4o-mini" #llama3.2

# --- NOVA FUNÇÃO HELPER: Para extrair texto com Tika ---
def extract_text_from_file(file: UploadFile) -> str:
    """
    Envia o conteúdo do arquivo para o endpoint /tika do Apache Tika e retorna o texto.
    """
    if not file:
        return ""
    
    # O endpoint /tika espera o conteúdo do arquivo no corpo da requisição
    # e o tipo de conteúdo no header 'Content-Type'.
    # Para extrair texto puro, definimos o header 'Accept'.
    headers = {
        "Accept": "text/plain",
        "Content-Type": file.content_type, # Passa o content-type original do arquivo
    }
    
    try:
        # Lê o conteúdo do arquivo e envia para o Tika
        content = file.file.read()
        response = requests.put(f"{TIKA_SERVER_ENDPOINT}/tika", headers=headers, data=content)
        
        response.raise_for_status() # Lança um erro para status 4xx ou 5xx
        return response.text
    except requests.RequestException as e:
        print(f"ERRO ao comunicar com o Tika: {e}")
        # Retorna uma string de erro que pode ser mostrada ao usuário
        return f"[ERRO: Não foi possível processar o anexo '{file.filename}'. Tika indisponível.]"
    except Exception as e:
        print(f"ERRO inesperado ao processar arquivo: {e}")
        return f"[ERRO: Falha ao ler o anexo '{file.filename}'.]"



# --- NOVO ENDPOINT: /find_series ---
@app.post("/find_series")
def find_series(req: FindRequest):
    """
    Etapa 1: Recebe uma pergunta e retorna uma lista de séries candidatas.
    """
    try:
        # Assumindo que você implementou a busca por código de série
        # A busca precisa retornar sercodigo, nome e score.
        # Você precisará ajustar sua função de busca no Redis para isso.
        series_found = store.knn_search_for_series_code(req.question, k=req.top_k)
        return {"series": series_found}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query")
def query(
        question: str = Form(...),
    sercodigo: str = Form(...),
    use_model: str = Form(...),
    model_name: str = Form(...),
    attachment: Optional[UploadFile] = File(None) # O anexo é opcional
):
    """
    Etapa 2: Recebe uma série confirmada, busca os dados, cria o contexto e consulta o LLM.
    """
    try:
        # ETAPA 1: Processar o anexo, se houver
        attachment_context = ""
        if attachment:
            print(f"Processando anexo: {attachment.filename}")
            attachment_context = extract_text_from_file(attachment)

        # 1. Buscar os dados completos da série no IPEA
        df = ip.timeseries(sercodigo)

        
        if 'YEAR' in df.columns and 'MONTH' in df.columns and 'DAY' in df.columns:
            # Constrói o índice de data a partir das colunas
            df.index = pd.to_datetime(df[['YEAR', 'MONTH', 'DAY']], errors='coerce')
        else:
            # Assume o "Formato A" e apenas garante que o índice é datetime
            df.index = pd.to_datetime(df.index, errors='coerce')
        
        # Remove quaisquer linhas onde a data não pôde ser convertida
        df = df[df.index.notna()]
        
        if df.empty:
            # Verificação antecipada para o caso de a série inteira ser vazia
            raise ValueError(f"A série {sercodigo} não retornou dados válidos do IPEA.")
            
        meta = ip.metadata(sercodigo)
        nome_serie = meta['NAME'].iloc[0] if not meta.empty else sercodigo

        # 2. Extrair o período de tempo da pergunta (lógica simples)
        anos = re.findall(r'\b(19|20)\d{2}\b', question)
        anos = sorted([int(ano) for ano in anos])
        
        print("\n--- INICIANDO ETAPA DE FILTRAGEM DE DATAS ---")

        if len(anos) >= 2:
            start_year, end_year = anos[0], anos[-1]
            print(f"Período de datas encontrado na pergunta: {start_year} a {end_year}")
            
            # Log do intervalo de datas do DataFrame ANTES do filtro
            print(f"Intervalo de datas do DataFrame original: {df.index.min().year} a {df.index.max().year}")
            
            # Aplica o filtro
            df_filtered = df[(df.index.year >= start_year) & (df.index.year <= end_year)]
            
            print(f"Tamanho do DataFrame original: {len(df)} linhas")
            print(f"Tamanho do DataFrame após o filtro: {len(df_filtered)} linhas")
        else:
            # Caso nenhum período de data seja encontrado na pergunta
            print("Nenhum período de datas válido (ex: 'entre 2010 e 2020') foi encontrado na pergunta.")
            print("Usando a série completa.")
            df_filtered = df

        print("--- FILTRAGEM DE DATAS CONCLUÍDA ---\n")

        # 3. Construção do contexto e resposta
        if df_filtered.empty:
            context = create_context_for_llm(df, nome_serie, sercodigo, question)
        else:
            context = create_context_for_llm(df_filtered, nome_serie, sercodigo, question)

        # ETAPA 3: Combinar os dois contextos em um só
        final_context = ""
        if attachment_context:
            final_context += "--- CONTEXTO DO DOCUMENTO ANEXADO ---\n"
            final_context += attachment_context
            final_context += "\n--- FIM DO DOCUMENTO ANEXADO ---\n\n"
        
        final_context += "--- CONTEXTO DA BASE DE DADOS IPEA ---\n"
        final_context += context
        final_context += "\n--- FIM DA BASE DE DADOS IPEA ---"

        # 4. Chamar o LLM com o novo contexto (sua lógica de chamada do LLM aqui)
        # llm_answer = call_your_llm_function(question, context, model_name)
        llm_answer = generate_answer(final_context)
        #llm_answer = f"Resposta simulada do LLM para a pergunta '{question}' sobre a série '{nome_serie}' com base nos dados fornecidos." # Placeholder

        # 5. Preparar dados para o gráfico
        df_chart_data = df_filtered.iloc[:, [-1]]
        df_chart_data = df_chart_data.reset_index()
        df_chart_data.columns = ['date', 'value']
        chart_data = df_chart_data.to_dict(orient='records')
        
        return {
            "llm_text": llm_answer,
            "context_used": context,
            "chart_data": chart_data,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def create_context_for_llm(df: pd.DataFrame, nome_serie: str, sercodigo: str, question: str) -> str:
    """Função auxiliar para criar o contexto em texto a partir do DataFrame."""
    
    # Usar .iloc[0] para pegar o valor da primeira coluna, qualquer que seja o nome
    initial_value = df.iloc[0, -1]
    final_value = df.iloc[-1, -1]
    initial_date = df.index[0].strftime('%Y-%m-%d')
    final_date = df.index[-1].strftime('%Y-%m-%d')

     # A lógica de resampling também usa a última coluna.
    resampled_df = df.iloc[:, [-1]].resample('A')
    resampled_df = resampled_df.mean()
    
    contexto = (
        f"Você é um assistente de análise de dados.\n\n"
        f'PERGUNTA DO USUÁRIO: "{question}"\n\n'
        f"Responda à pergunta do usuário com base no seguinte contexto:\n\n"
        f"Série: {nome_serie} ({sercodigo})\n"
        f"Período de Análise: de {initial_date} a {final_date}\n\n"
        "Resumo da Evolução:\n"
        f"- Valor Inicial: {initial_value:.2f} em {initial_date}\n"
        f"- Valor Final: {final_value:.2f} em {final_date}\n\n"
        "Dados Pontuais (amostra):\n"
        # Usar to_markdown para uma tabela bem formatada que o LLM entende bem
        f"{resampled_df.to_markdown()}"
    )
    return contexto

# --- NOVO ENDPOINT PARA OBTER SÉRIES INDEXADAS ---
@app.get("/indexed_series")
def get_indexed_series():
    """
    Replica a lógica do script bash para obter os códigos únicos das séries
    indexadas no Redis e os enriquece com seus nomes.
    """
    try:
        # 1. Obter todos os códigos únicos do Redis (equivalente a `redis-cli --scan | sed | sort -u`)
        # Usar um set garante a unicidade e é muito eficiente.
        indexed_codes = set()
        # O prefixo deve corresponder ao que você usa no RedisVectorStore
        for key in store.r.scan_iter("doc:ipea:*"):
            # Ex: 'doc:ipea:ABATE_ABPEAV:1970-01-01T00:00:00' -> 'ABATE_ABPEAV'
            parts = key.decode('utf-8').split(':')
            if len(parts) > 2:
                indexed_codes.add(parts[2])

        if not indexed_codes:
            return {"series": [], "total": 0}

        # 2. Usar o DataFrame de metadados em cache para obter os nomes
        metadata_df = app_state.get("metadata_df")
        if metadata_df is None or metadata_df.empty:
             raise HTTPException(status_code=500, detail="Cache de metadados não está disponível.")

        # Filtra o DataFrame para conter apenas as séries que encontramos no Redis
        filtered_df = metadata_df[metadata_df['CODE'].isin(indexed_codes)]

        # 3. Formata os dados para a resposta JSON
        result_df = filtered_df[['CODE', 'NAME']].copy()
        result_df.rename(columns={'CODE': 'sercodigo', 'NAME': 'nome'}, inplace=True)
        
        series_list = result_df.to_dict(orient='records')

        return {"series": series_list, "total": len(series_list)}

    except Exception as e:
        print(f"ERRO em /indexed_series: {e}")
        raise HTTPException(status_code=500, detail=str(e))