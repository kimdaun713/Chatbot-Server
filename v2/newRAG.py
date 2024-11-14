import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import sqlite3
from fastapi import FastAPI
from vllm import LLM, SamplingParams
import uvicorn

app = FastAPI()

# 임베딩 모델 로드
embedding_model = SentenceTransformer("jhgan/ko-sroberta-multitask")

# LLM 모델 로드
model_path = '/home/sslab/LLM/Models/EEVE-Korean-Instruct-10.8B-v1.0'
llm = LLM(model=model_path, max_model_len=2400)

# 저장된 FAISS 인덱스 로드
faiss_index = faiss.read_index("faiss_hnsw_index.idx")

# SQLite 데이터베이스에서 ID로 문서 가져오기
def get_docs_by_ids(id_list):
    try:
        # 데이터베이스 연결
        conn = sqlite3.connect("docs.db")
        conn.row_factory = sqlite3.Row  # 행을 딕셔너리 형태로 반환
        cursor = conn.cursor()

        # FAISS 인덱스의 ID는 0부터 시작하므로, 데이터베이스의 ID와 맞추기 위해 +1
        id_list = [id + 1 for id in id_list]
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

from pydantic import BaseModel
import logging

logging.basicConfig(level=logging.INFO)


class PromptRequest(BaseModel):
    prompt: str

# 유사한 문서 검색 함수
def search_similar_docs(query, index, model, top_k=5):
    # 검색 쿼리를 임베딩 벡터로 변환
    query_embedding = model.encode(query)

    # FAISS로 유사한 항목 검색
    D, I = index.search(np.array([query_embedding]), top_k)
    id_list = I[0].tolist()
    logging.info(f"relevant: {id_list}")
    # 해당 ID의 문서들을 데이터베이스에서 가져오기
    result = get_docs_by_ids(id_list)
    return result

# API 엔드포인트 정의
@app.post("/query")
async def query_api(request: PromptRequest):
    # 1. RAG를 통해 관련 문서 검색
    similar_docs = search_similar_docs(request.prompt, faiss_index, embedding_model, top_k=5)
    # 2. 프롬프트와 검색된 문서 결합
    context = similar_docs
    
    full_prompt = f"""
    
    당신은 인공지능 어시스트입니다.  
    계산 과정이 포함된 경우 답변에 계산 과정을 포함하지 않고 간결하게 답만 한국어로 답해주세요. \n
    iHuman: 계산이 필요한 질문일 경우 계산 과정을 포함하지 않고 간결하게 답만 한국어로 답해주세요. 
    Context를 우선 참고해서 말해주세요. 관련없는 Context라면 참고해서 말하지 마세요.\n
    Context: {context}\n
    Question:  {request.prompt}\n
    Assistant:\n

    """
    logging.info(context)
    # 3. LLM을 사용하여 응답 생성
    sampling_params = SamplingParams(max_tokens=512, temperature=0.2)
    #sampling_params = BeamSearchParams(beam_width=3, max_tokens = 512)
    outputs = llm.generate(full_prompt, sampling_params)
    response = ""
    for output in outputs:
        response = response + output.outputs[0].text

    # 응답 반환
    return {"response": response}

# 서버 실행 (필요한 경우)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

