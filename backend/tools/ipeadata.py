# backend/tools/ipeadata.py
import requests
import re
import time
from typing import Optional, Dict, Any, List
from urllib.parse import quote_plus

BASE_ODATA = "https://www.ipeadata.gov.br/api/odata4"

def get_with_retries(url, params, retries=3, delay=5, timeout=60):
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
            if attempt < retries - 1:
                time.sleep(delay)
                continue
            else:
                raise

def get_series_values(sercodigo: str, top: int = 1000) -> List[Dict[str, Any]]:
    """
    Fetch series values for a given SERCODIGO (series code).
    Example: ValoresSerie(SERCODIGO='PRECOS12_IPCA12')
    Returns list of {Data, Valor, ...}
    """
    # Build OData function call
    func = f"ValoresSerie(SERCODIGO='{sercodigo}')"
    url = f"{BASE_ODATA}/{func}"
    resp = get_with_retries(url, params={"top": top})
    resp.raise_for_status()
    js = resp.json()
    # OData returns 'value' typically
    return js.get("value", []) if isinstance(js, dict) else js

def search_metadata_by_keyword(q: str, top: int = 20) -> List[Dict[str, Any]]:
    """
    Search Metadados by keyword in Portuguese (naive).
    Uses OData filter $filter=contains(Descricao,'q') or Title etc.
    """
    """
    Naive search over Metadados using OData contains on Nome/Descricao/SERCODIGO.
    """
    if not q:
        return []
    # Escape q for URL
    q_esc = q.replace("'", "''").lower()
    q_esc = re.sub(r"[^\w\s]", "", q_esc)  # remove pontuação
    q_esc = q_esc.strip()
    # Use contains on Descricao or SERCODIGO or Nome
    filter_query = (
        f"contains(tolower(Descricao),'{q_esc}') or "
        f"contains(tolower(Nome),'{q_esc}') or "
        f"contains(tolower(SERCODIGO),'{q_esc}')"
    )
    url = f"{BASE_ODATA}/Metadados"
    resp = requests.get(url, params={"filter": filter_query, "top": top}, timeout=30)
    resp.raise_for_status()
    js = resp.json()
    return js.get("value", []) if isinstance(js, dict) else js

def get_metadata_by_sercodigo(sercodigo: str) -> Optional[Dict[str, Any]]:
    """
    Get metadata for a specific SERCODIGO
    """
    url = f"{BASE_ODATA}/Metadados"
    # OData filter
    params = {"$filter": f"SERCODIGO eq '{sercodigo}'", "$top": 1}
    resp = requests.get(url, params=params,timeout=20)
    resp.raise_for_status()
    js = resp.json()
    vals = js.get("value", [])
    return vals[0] if vals else None