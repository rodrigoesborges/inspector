# backend/rag/embedding.py
from sentence_transformers import SentenceTransformer
import numpy as np
import redis
import os
import time
from typing import List, Dict, Any, Optional

# Importar classes necessárias para a busca em Redis
from redis.commands.search.field import TextField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query as RediSearchQuery
from redis.exceptions import ResponseError, BusyLoadingError

MODEL_NAME = os.environ.get("SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")
EMBED_DIM = int(os.environ.get("EMBED_DIM", "384"))  # matches all-MiniLM-L6-v2
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:8999")
INDEX_NAME = os.environ.get("REDIS_INDEX_NAME", "idx:ipea")
DOC_PREFIX = os.environ.get("REDIS_DOC_PREFIX", "doc:ipea:")

class RedisVectorStore:
    def __init__(self, redis_url: str = REDIS_URL):
        self.r = redis.Redis.from_url(redis_url)
        self.model = SentenceTransformer(MODEL_NAME)
        #self._ensure_index()
        # --- CORREÇÃO AQUI ---
        # Adiciona um loop de retentativa para esperar o Redis ficar pronto.
        
        max_retries = 30  # Tenta por no máximo 30 segundos
        retries = 0
        
        while retries < max_retries:
            try:
                # Tenta inicializar o índice. Se funcionar, sai do loop.
                self._ensure_index()
                print("✅ Conexão com Redis e índice verificados com sucesso.")
                break 
            except BusyLoadingError:
                # Se o Redis estiver ocupado, espera 1 segundo e tenta novamente.
                retries += 1
                print(f"⏳ Redis está carregando... aguardando 1s. (Tentativa {retries}/{max_retries})")
                time.sleep(10)
            except Exception as e:
                # Se outro erro ocorrer, falha imediatamente.
                print(f"❌ Erro inesperado ao conectar com o Redis: {e}")
                raise e # Lança a exceção original para parar a aplicação
                
        if retries == max_retries:
            raise Exception("❌ O Redis não ficou pronto a tempo. A aplicação será encerrada.")

    
    def _ensure_index(self):
        try:
            self.r.ft(INDEX_NAME).info()
        except ResponseError:
            # Cria o índice se ele não existir
            schema = (
                TextField("text"),
                TextField("sercodigo"),
                TextField("date"),
                TextField("value"),
                VectorField("vector", "FLAT", {
                    "TYPE": "FLOAT32",
                    "DIM": EMBED_DIM,
                    "DISTANCE_METRIC": "COSINE"
                })
            )
            definition = IndexDefinition(prefix=[DOC_PREFIX], index_type=IndexType.HASH)
            self.r.ft(INDEX_NAME).create_index(
                fields=list(schema),
                definition=definition
            )

    def _to_bytes(self, arr: np.ndarray) -> bytes:
        return arr.astype(np.float32).tobytes()

    def _from_bytes(self, b: bytes) -> np.ndarray:
        return np.frombuffer(b, dtype=np.float32)

    def embed(self, text: str) -> np.ndarray:
        # Simplificado, pois o modelo já produz a dimensão correta.
        return self.model.encode([text], convert_to_numpy=True)[0]

    def add_doc(self, id_: str, text: str, meta: Dict[str, Any]):
        vec = self.embed(text)
        key = f"{DOC_PREFIX}{id_}"
        mapping = {"text": text}
        mapping.update({k: str(v) for k, v in meta.items()})
        mapping["vector"] = self._to_bytes(vec)
        self.r.hset(key, mapping=mapping)

    def knn_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        q_vec = self.embed(query)
        q_bytes = self._to_bytes(q_vec)
        
        # A consulta deve ser uma string no formato do RediSearch
        base_q = f"*=>[KNN {k} @vector $vec AS score]"
        
        # Usar o objeto Query do redis-py para construir a busca
        res = self.r.ft(INDEX_NAME).search(
            RediSearchQuery(base_q)
            .return_fields("text", "sercodigo", "date", "value", "score")
            .dialect(2),
            query_params={"vec": q_bytes}
        )
        
        out = []
        # Corrigido o loop para extrair os resultados corretamente
        for doc in res.docs:
            d: Dict[str, Any] = {
                "text": getattr(doc, "text", None),
                "sercodigo": getattr(doc, "sercodigo", None),
                "date": getattr(doc, "date", None),
                "value": getattr(doc, "value", None),
                "score": float(getattr(doc, "score", 0))
            }
            out.append(d)
        return out

    def knn_search_for_series_code(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Busca as SÉRIES mais relevantes para uma pergunta, retornando códigos únicos.

        Esta função é otimizada para a ETAPA 1 do fluxo de RAG de duas etapas.
        Ela busca mais resultados do que o 'k' solicitado e depois os agrupa
        para garantir que retornemos 'k' séries *distintas*.

        Args:
            query (str): A pergunta do usuário.
            k (int): O número de séries únicas a serem retornadas.

        Returns:
            List[Dict[str, Any]]: Uma lista de dicionários, cada um contendo
                                  'sercodigo', 'nome', e 'score'.
        """
        # 1. Embedar a pergunta do usuário (nenhuma mudança aqui)
        q_vec = self.embed(query)
        q_bytes = self._to_bytes(q_vec)

        # 2. Construir a consulta de busca por vetor
        # Heurística: buscamos mais resultados (k * 10) para aumentar a
        # chance de encontrar k séries *únicas* entre os resultados.
        num_results_to_fetch = k * 10
        base_q = f"*=>[KNN {num_results_to_fetch} @vector $vec AS score]"

        # 3. Definir a consulta e os campos a serem retornados
        # ESTA É A MUDANÇA PRINCIPAL: Pedimos 'sercodigo', 'nome' e 'score'.
        query_obj = (
            RediSearchQuery(base_q)
            .sort_by("score")
            .return_fields("sercodigo", "nome", "score") # <<-- AQUI ESTÁ A MÁGICA
            .dialect(2)
        )

        # 4. Executar a busca
        try:
            res = self.r.ft(INDEX_NAME).search(
                query_obj,
                query_params={"vec": q_bytes}
            )
        except Exception as e:
            print(f"Erro durante a busca no Redis: {e}")
            return []

        # 5. Processar e desduplicar os resultados
        # Isso é crucial, pois os N primeiros resultados podem ser da mesma série.
        series_found = {}
        for doc in res.docs:
            code = doc.sercodigo
            score = float(doc.score)
            
            # Se já encontramos k séries únicas e esta é uma nova, podemos ignorar.
            if len(series_found) >= k and code not in series_found:
                continue

            # Adiciona a série se for nova ou se o score for melhor (menor) que o já salvo.
            if code not in series_found or score < series_found[code]['score']:
                series_found[code] = {
                    "sercodigo": code,
                    "nome": doc.nome,
                    "score": score
                }
        
        # 6. Formatar a saída final
        # Converte o dicionário de séries únicas em uma lista e a ordena pelo score.
        final_list = sorted(list(series_found.values()), key=lambda x: x['score'])
        
        # Garante que retornemos no máximo k resultados
        return final_list[:k]
