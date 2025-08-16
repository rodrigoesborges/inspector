# backend/tools/ipeadata.py
import requests
import os
import re
import time
import redis
import pandas as pd
import ipeadatapy as ip
import json  # CORREÇÃO: Adicionada importação do módulo json
from typing import Optional, Dict, Any, List, Tuple
from urllib.parse import quote_plus

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:8999")
r = redis.Redis.from_url(REDIS_URL)

def get_series_values(
    sercodigo: str, 
    start_date: Optional[str] = None,  # CORREÇÃO: Adicionados argumentos ausentes
    end_date: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Fetch series values for a given SERCODIGO (series code).
    Example: ValoresSerie(SERCODIGO='PRECOS12_IPCA12')
     Retorna um DataFrame pronto para análise.
    """
    try:
        df = ip.timeseries(sercodigo)
    except Exception as e:
        # Se a série não for encontrada, retorna um DataFrame vazio
        print(f"Aviso: Falha ao buscar a série {sercodigo}. Erro: {e}")
        return pd.DataFrame()

    if df.empty:
        return df

    # 1. Garante que o índice é do tipo DatetimeIndex
    df.index = pd.to_datetime(df.index)

    # 2. Filtra pelo período usando o índice Datetime (forma correta)
    if start_year and end_year:
        # O filtro agora é seguro e eficiente
        df = df[(df.index.year >= start_year) & (df.index.year <= end_year)]
    
    # 3. Retorna o DataFrame filtrado (não um dicionário!)
    return df

def search_metadata_by_keyword(q: str, top: int = 20) -> List[Dict[str, Any]]:
    """
    Search Metadados by keyword in Portuguese (naive).
    Uses OData filter $filter=contains(Descricao,'q') or Title etc.
    Naive search over Metadados using OData contains on
    Nome/Descricao/SERCODIGO.
    """
    if not q:
        return []

    results = []
    q_lower = q.lower()  # CORREÇÃO: Usar a variável `q`

    # CORREÇÃO: Usar `scan_iter` para um método de busca mais seguro e performático
    for key in r.scan_iter("meta:*"):
        meta = json.loads(r.get(key))
        if any(q_lower in str(v).lower() for v in meta.values()):
            results.append(meta)
            if len(results) >= top:
                break
    return results

def knn_search_for_series_code(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
    """Busca apenas pelos códigos de série mais relevantes."""
    q_vec = self.embed(query)
    q_bytes = self._to_bytes(q_vec)
    
    base_q = f"*=>[KNN {k} @vector $vec AS score]"
    
    # IMPORTANTE: Peça para retornar apenas o 'sercodigo' e o score.
    # Usamos .limit(0, k) para pegar apenas os k primeiros resultados.
    res = self.r.ft(self.index_name).search(
        RediSearchQuery(base_q)
        .return_fields("sercodigo", "score")
        .sort_by("score")
        .dialect(2),
        query_params={"vec": q_bytes}
    )
    
    # Agrupa para retornar códigos únicos e a melhor pontuação para cada um
    series_found = {}
    for doc in res.docs:
        code = doc.sercodigo
        score = float(doc.score)
        if code not in series_found or score < series_found[code]['score']:
             series_found[code] = {"sercodigo": code, "score": score}
             
    return list(series_found.values())


def get_metadata_by_sercodigo(sercodigo: str) -> Optional[Dict[str, Any]]:
    """
    Get metadata for a specific SERCODIGO
    """
    meta = r.get(f"meta:{sercodigo}")
    if meta:
        return json.loads(meta)

    # Busca direta do IPEA caso não esteja no Redis
    # CORREÇÃO: Usar a variável `ip` do import `ipeadatapy as ip`
    metadata_df = ip.metadata()
    meta_row = metadata_df[metadata_df["CODE"] == sercodigo]
    if not meta_row.empty:
        meta_json = meta_row.iloc[0].to_json()
        r.set(f"meta:{sercodigo}", meta_json)
        return json.loads(meta_json)
    return None
