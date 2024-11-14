from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.retrievers import MultiQueryRetriever
import faiss
import numpy as np
import sqlite3
from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

# SQLite 데이터베이스에서 데이터 가져오기
def fetch_data_from_sqlite(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT id, doc FROM docs")  # 예시로 id와 텍스트 가져오기
    data = cursor.fetchall()
    conn.close()
    return data

# 임베딩 모델 로드
embedder = SentenceTransformer("jhgan/ko-sroberta-multitask")
embedding_model = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")

# 데이터 임베딩 함수
def embed_data(data):
    texts = [record[1] for record in data]
    embeddings = embedder.encode(texts, convert_to_tensor=True)
    return embeddings

# FAISS 인덱스 생성
def create_faiss_index(embeddings):
    d = embeddings.shape[1]  # 벡터 차원
    index = faiss.IndexHNSWFlat(d, 32)  # HNSW 인덱스, efConstruction=32
    index.hnsw.efSearch = 64  # 검색 성능 조정
    index.add(embeddings.cpu().numpy())  # 임베딩만 추가 (ID 없이 추가)
    return index

# 검색 쿼리 처리 함수
def search_similar_docs(query, index, model, data, top_k=5):
    # 검색 쿼리를 임베딩 벡터로 변환하고 CPU로 이동
    query_embedding = model.encode(query, convert_to_tensor=True).cpu().numpy()

    # query_embedding을 2차원 배열로 변환하여 검색
    D, I = index.search(np.array([query_embedding]), k=top_k)  # top_k개의 유사한 항목 반환

    # 검색 결과로부터 데이터 ID 추출
    results = [data[idx] for idx in I[0]]  # 상위 top_k의 인덱스에 해당하는 데이터
    return results

# 데이터 가져오기 및 임베딩
data = fetch_data_from_sqlite("docs.db")
embeddings = embed_data(data)

# FAISS 인덱스 생성 및 로컬에 저장
faiss_index = create_faiss_index(embeddings)

faiss.write_index(faiss_index, "deposit_index.idx")

# docstore: 임베딩에 대한 메타데이터를 저장하는 임시 딕셔너리 (예시)
docstore = {record[0]: record[1] for record in data}

# index_to_docstore_id: 인덱스와 문서 ID를 매핑하는 함수
def index_to_docstore_id(index):
    return docstore.get(index)

# FAISS 인덱스와 docstore, index_to_docstore_id를 전달하여 vector_store 생성
vector_store = FAISS(
    index=faiss_index,
    embedding_function=embedding_model,
    docstore=docstore,
    index_to_docstore_id=index_to_docstore_id
)


# MultiQueryRetriever 생성
retriever = MultiQueryRetriever(base_retriever=vector_store)

# 유사한 인덱스를 반환하는 함수
def retrieve_indices(query, retriever, top_k=5):
    results = retriever.retrieve([query])  # MultiQueryRetriever를 통해 검색

    # 검색된 결과에서 유사한 인덱스만 추출 (문서 ID만 추출)
    indices = []
    for result in results:
        print(result)  # 결과 형식 확인 (디버깅용)
        if isinstance(result, dict):
            indices.extend(result.get('results', []))  # 검색 결과에서 'results' 추출

    return indices[:top_k]

# 여러 쿼리 검색 예시
queries = ["최고 금리가 있는 예금", "최고의 예금 상품"]
for query in queries:
    similar_indices = retrieve_indices(query, retriever)
    print(f"Query: {query}, Similar Indices: {similar_indices}")

