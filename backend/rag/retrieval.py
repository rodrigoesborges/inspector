# backend/rag/retrieval.py
from .embedding import RedisVectorStore #FAISS Embedder, FaissStore
from typing import List, Dict, Any
import numpy as np
import textwrap

#embedder = Embedder()
#store = FaissStore()
store = RedisVectorStore()

def index_ipea_series(sercodigo: str, series_values: List[Dict[str, Any]]):
    """
    FAISS
    Convert series values into text chunks, embed and add to store.
    Each meta: {'sercodigo': ..., 'date': ..., 'value': ..., 'text': ...}
    chunks = []
    metas = []
    """
    """
    Index each row from IPEA series into Redis vector index.
    """
    count = 0
    for i, row in enumerate(series_values):
        # typical row may have 'Data' and 'Valor'
        date = row.get("Data") or row.get("DataReferencia") # FAISS or row.get("data")
        val = row.get("Valor") # FAISS or row.get("valor") or row.get("ValorNumerico") or row.get("ValorTexto")
        text = f"Série {sercodigo} — Data: {date} — Valor: {val}"
        #FAISS chunks.append(text)
        meta = {"sercodigo": sercodigo, "date": date or "", "value": str(val or "")} #FAISS metas.append(..., "text": text)
        store.add_doc(f"{sercodigo}:{i}",text,meta)
        count +=1
    return count
#    return len(series_values)
'''
   #FAISS     
    if not chunks:
        return 0
    vecs = embedder.encode(chunks)
    store.add(vecs, metas)
    return len(chunks)
'''


def retrieve_similar(query: str, k: int = 5): #FAISS -> List[Dict[str, Any]]:
    return store.knn_search(query, k=k)
'''
#FAIS
    qv = embedder.encode([query])
    results = store.search(qv, k=k)
    # results: list of (dist, meta)
    return [{"score": r[0], **r[1]} for r in results]
'''

def build_context_from_results(results: List[Dict[str, Any]], max_chars: int = 2000) -> str:
    pieces = []
    for r in results:
        pieces.append(r.get("text", ""))
    ctx = "\n".join(pieces)
    if len(ctx) > max_chars:
        return textwrap.shorten(ctx, width=max_chars, placeholder=" ...")
    return ctx
