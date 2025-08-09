# backend/rag/embedding.py
from sentence_transformers import SentenceTransformer
import numpy as np
#import faiss
import redis
import os
import pickle
from redis.exceptions import ResponseError
from typing import List, Tuple, Dict

MODEL_NAME = os.environ.get("SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")
EMBED_DIM = int(os.environ.get("EMBED_DIM", "384"))  # matches all-MiniLM-L6-v2
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:8999")
INDEX_NAME = os.environ.get("REDIS_INDEX_NAME", "idx:ipea")
DOC_PREFIX = os.environ.get("REDIS_DOC_PREFIX", "doc:ipea:")

'''
class FaissStore:
    def __init__(self, dim=EMBED_DIM, index_path: str = "faiss.index", meta_path: str = "faiss_meta.pkl"):
        self.dim = dim
        self.index_path = index_path
        self.meta_path = meta_path
        if os.path.exists(index_path) and os.path.exists(meta_path):
            self.index = faiss.read_index(index_path)
            with open(meta_path, "rb") as f:
                self.meta = pickle.load(f)
        else:
            self.index = faiss.IndexFlatL2(dim)
            self.meta = []  # list of metadata dicts for each vector

    def add(self, vectors: np.ndarray, metas: List[Dict]):
        assert vectors.shape[1] == self.dim
        self.index.add(vectors.astype("float32"))
        self.meta.extend(metas)
        self._persist()

    def search(self, query_vec: np.ndarray, k: int = 5) -> List[Tuple[float, Dict]]:
        D, I = self.index.search(query_vec.astype("float32"), k)
        results = []
        for dist, idx in zip(D[0], I[0]):
            if idx < len(self.meta):
                results.append((float(dist), self.meta[idx]))
        return results

    def _persist(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.meta, f)
'''

class RedisVectorStore:
    def __init__(self, redis_url: str = REDIS_URL):
        self.r = redis.Redis.from_url(redis_url)
        #self.embedder = SentenceTransformer(MODEL_NAME)
        self.model = SentenceTransformer(MODEL_NAME)
        self._ensure_index()

    def _ensure_index(self):
        try:
            _ = self.r.ft(INDEX_NAME).info()
        except ResponseError:
            # Create index
            from redis.commands.search.field import TextField, VectorField
            from redis.commands.search.index_definition import IndexDefinition, IndexType

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
        vec = self.model.encode([text], convert_to_numpy=True)[0]
        if vec.shape[0] != EMBED_DIM:
            # try to handle mismatch gracefully
            vec = vec[:DIM] if vec.shape[0] > DIM else np.pad(vec, (0, DIM - vec.shape[0]))
        return vec

    def add_doc(self, id_: str, text: str, meta: dict):
        vec = self.embed(text)
        key = f"{DOC_PREFIX}{id_}"
        mapping = {"text": text}
        mapping.update({k: str(v) for k, v in meta.items()})
        mapping["vector"] = self._to_bytes(vec)
        self.r.hset(key, mapping=mapping)

    def knn_search(self, query: str, k: int = 5):
        q_vec = self.embed(query)
        q_bytes = self._to_bytes(q_vec)
        base_q = f"*=>[KNN {k} @vector $vec AS score]"
        q = self.r.ft(INDEX_NAME).search(
            redis.commands.search.query.Query(base_q)
            .return_fields("text", "sercodigo", "date", "value", "score")
            .dialect(2),
            query_params={"vec": q_bytes}
        )
        out = []
        for doc in q.docs:
            d = {}
        for f in ("text", "sercodigo", "date", "value"):
            d[f] = getattr(doc, f, None)            
            # include score if present
            if hasattr(doc, "score"):
                d["score"] = doc.score
            out.append(d)
        return out

''' FAISS
    def embed(self, text: str) -> bytes:
        vec = self.embedder.encode([text])[0].astype(np.float32)
        return vec.tobytes()

    def add(self, id_: str, text: str, meta: dict):
        vec_bytes = self.embed(text)
        key = f"{DOC_PREFIX}{id_}"
        self.r.hset(key, mapping={**meta, "text": text, "vector": vec_bytes})

    def search(self, query: str, k: int = 5):
        q_vec = self.embed(query)
        base_query = f"*=>[KNN {k} @vector $vec AS score]"
        res = self.r.ft(INDEX_NAME).search(
            redis.commands.search.query.Query(base_query)
            .sort_by("score")
            .return_fields("text", "score", "sercodigo", "date", "value")
            .dialect(2),
            query_params={"vec": q_vec}
        )
        return [doc.__dict__ for doc in res.docs]

class Embedder:
    def __init__(self, model_name=MODEL_NAME):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

'''

