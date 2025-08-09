# backend/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from tools.ipeadata import search_metadata_by_keyword, get_series_values, get_metadata_by_sercodigo
from rag.retrieval import index_ipea_series, retrieve_similar, build_context_from_results
from llm.ollama_client import generate_answer
import uvicorn
import os

app = FastAPI(title="IPEADATA-RAG-Redis-Backend-POC")

class QueryRequest(BaseModel):
    question: str
    sercodigo: Optional[str] = None   # optional direct series code
    top_k: Optional[int] = 5
    use_model: Optional[str] = "ollama"  # or 'openai'
    model_name: Optional[str] = "gpt-5-mini" #llama3.2

@app.post("/query")
def query_endpoint(req: QueryRequest):
    # Step 1: if sercodigo provided, fetch series values and index them
    if req.sercodigo:
        try:
            values = get_series_values(req.sercodigo, top=2000)
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"IPEA fetch error: {e}")
        indexed = index_ipea_series(req.sercodigo, values)
        indexed_note = f"Indexed {indexed} rows from series {req.sercodigo}.\n"
    else:
        indexed_note = ""

    # Step 2: If no series provided, try to find candidate series by keyword
    if not req.sercodigo:
        candidates = search_metadata_by_keyword(req.question, top=5)
        if candidates:
            # choose first candidate for indexing (naive)
            ser = candidates[0].get("SERCODIGO")
            if ser:
                values = get_series_values(ser, top=2000)
                index_ipea_series(ser, values)
                indexed_note = f"Found and indexed series {ser} (best match).\n"

    # Step 3: Retrieve the top-k similar chunks from the vector DB
    results = retrieve_similar(req.question, k=req.top_k)
    context = build_context_from_results(results, max_chars=3000)

    # Step 4: Build the prompt for LLM
    system = (
        "Você é um assistente que responde perguntas sobre séries do IPEA.\n"
        "Use estritamente os dados recuperados abaixo para formular uma resposta em Português.\n"
        "Se não houver informação suficiente, diga que não é possível responder com os dados disponíveis.\n\n"
    )
    prompt = (
        system
        + "Contexto (trechos recuperados):\n"
        + context
        + "\n\nPergunta do usuário:\n"
        + req.question
        + "\n\nResponda de forma sucinta e cite as datas/valores quando possível.\n"
    )

    # Step 5: Call LLM
    try:
        llm_resp = generate_answer(prompt, preferred=req.use_model, model=req.model_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {e}")

    return {
        "indexed_note": indexed_note,
        "retrieval_hits": results,
        "llm_text": llm_resp.get("text"),
        "llm_raw": llm_resp.get("raw")
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("API_PORT", 8997)))
