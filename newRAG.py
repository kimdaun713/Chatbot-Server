import sqlite3
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch
import torch
from vllm import LLM, SamplingParams

# SQLite 데이터베이스에서 데이터 가져오기
def fetch_data_from_sqlite(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM docs")  # 예시로 id와 텍스트 가져오기
    data = cursor.fetchall()
    #print(data)
    conn.close()
    return data

# 새로운 Ko-SRoBERTa 모델 설정
model = SentenceTransformer("jhgan/ko-sroberta-multitask")
#llm MODEL
model_path = '/home/sslab/LLM/Models/EEVE-Korean-Instruct-10.8B-v1.0'
llm = LLM(model=model_path, max_model_len=2400)



# 텍스트 데이터를 임베딩으로 변환
def embed_data(data):
    texts = [f"{record[0]}: {record[1]}" for record in data]
    embeddings = model.encode(texts)
    return embeddings

# FAISS 인덱스 생성 및 HNSW 설정
def create_faiss_index(embeddings):
    d = embeddings.shape[1]  # 벡터 차원
    index = faiss.IndexHNSWFlat(d, 64)  # HNSW 인덱스, efConstruction=32
    index.hnsw.efSearch = 128  # 검색 성능 조정
    
    index.add(embeddings)
    return index


def get_docs_by_ids(id_list):
    try:
        # 데이터베이스 연결
        conn = sqlite3.connect("docs.db")
        conn.row_factory = sqlite3.Row  # 행을 딕셔너리 형태로 반환
        cursor = conn.cursor()
        id_list = [id + 1 for id in id_list]
        print(id_list)
        # id 목록에 해당하는 문서들을 가져오는 쿼리 실행
        placeholders = ','.join('?' for _ in id_list)
        query = f"SELECT * FROM docs WHERE id IN ({placeholders})"
        cursor.execute(query, id_list)
        rows = cursor.fetchall()

        # 연결 닫기
        conn.close()

        # 결과를 문자열로 합치기
        result_string = "\n".join(" | ".join(f"{key}: {value}" for key, value in dict(row).items()) for row in rows)

        return result_string

    except sqlite3.Error as e:
        print(f"SQLite3 에러 발생: {e}")
        return None

# 검색 함수: 코사인 유사도 기반 검색
def search_similar_docs(query, index, model, data, embeddings, top_k=5):
    # 검색 쿼리를 임베딩 벡터로 변환
    query_embedding = model.encode(query)

    # FAISS로 유사한 항목 검색 (HNSW 기반)
    D, I = index.search(np.array([query_embedding]), top_k)
    id_list = I[0].tolist()
    print(D)
    print(I)
    result = get_docs_by_ids(id_list)
    return result

#def get_docs(I):

# SQLite 데이터 가져오기
data = fetch_data_from_sqlite("docs.db")
embeddings = embed_data(data)

# FAISS 인덱스 생성 및 로컬에 저장
faiss_index = create_faiss_index(embeddings)
faiss.write_index(faiss_index, "faiss_hnsw_index.idx")

# 유사한 상품 검색 예시
query = "id 30"
similar_docs = search_similar_docs(query, faiss_index, model, data, embeddings, top_k=5)

# 결과 출력
text = ''.join(similar_docs)
print(text)


