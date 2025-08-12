import ipeadatapy as ip
from rag.embedding import RedisVectorStore
import pandas as pd

def index_all_series():
    store = RedisVectorStore()

    print("Obtendo lista de todas as séries do IPEA...")
    meta_df = ip.metadata()  # DataFrame com CODE, NAME, UNIT, COMMENT, etc.
    all_codes = meta_df["CODE"].dropna().unique().tolist()
    print(f"✅ {len(all_codes)} códigos encontrados.")

    total_series = 0
    total_records = 0

    # Adicionando um contador de progresso
    from tqdm import tqdm

    for ser_code in tqdm(all_codes, desc="Indexando séries"):
        # OTIMIZAÇÃO 1: Buscar metadados apenas uma vez por série
        meta_row = meta_df[meta_df["CODE"] == ser_code]
        if meta_row.empty:
            continue
        
        meta_data = {
            "sercodigo": ser_code,
            "nome": meta_row["NAME"].values[0],
            "unidade": meta_row["UNIT"].values[0],
            "descricao": meta_row["COMMENT"].values[0]
        }
        
        print(f"Baixando série {ser_code}...")
        try:
            df = ip.timeseries(ser_code)
        except Exception as e:
            print(f"Erro na série {ser_code}: {e}")
            continue

        if df.empty:
            continue
            
        total_series += 1
        total_records += len(df)
        
        # OTIMIZAÇÃO 2: Preparar todos os documentos antes de indexar
        docs_to_add = []
        for i, row in df.iterrows():
            # Inclusão de Nome e Descrição no campo `text` para melhor RAG
            text = (
                f"Série {meta_data['nome']} ({ser_code}). "
                f"Descrição: {meta_data['descricao']}. "
                f"Data: {row.name.strftime('%Y-%m-%d')} - Valor: {row.iloc[-1]} ({meta_data['unidade']})"
            )
            
            # Assegurar que os metadados sejam strings para o Redis
            meta_for_redis = {
                "sercodigo": meta_data['sercodigo'],
                "date": str(row.name.strftime('%Y-%m-%d')),
                "value": row.iloc[-1],
                "nome": meta_data['nome'],
                "unidade": meta_data['unidade']
            }
            
            # Prepara o documento para a indexação
            docs_to_add.append({
                "id": f"{ser_code}:{i}",
                "text": text,
                "meta": meta_for_redis
            })

        # OTIMIZAÇÃO 3: Usar o método `add_docs` se disponível, ou iterar de forma mais eficiente
        if hasattr(store, 'add_docs'):
            store.add_docs(docs_to_add)
        else:
            # Alternativa mais rápida: usar hset multi-chave diretamente
            # Esta lógica pode precisar de ajuste dependendo de como a sua classe RedisVectorStore
            # foi implementada para lidar com múltiplos documentos
            for doc in docs_to_add:
                 store.add_doc(doc['id'], doc['text'], doc['meta'])

    print(f"\n✅ Indexação completa: {total_series} séries, {total_records} registros.")

if __name__ == "__main__":
    index_all_series()
