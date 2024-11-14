import sqlite3
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from langchain.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer
import torch

model_name = 'yanolja/EEVE-Korean-Instruct-10.8B-v1.0'

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config)
tokenizer = AutoTokenizer.from_pretrained(model_name)

text_generation_pipeline = pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    temperature=0.2,
    do_sample=True,
    repetition_penalty=1.1,
    return_full_text=True,
    max_new_tokens=400,
)

llm = HuggingFacePipeline(pipeline=text_generation_pipeline)



prompt_template = """
당신은 인공지능 어시스트입니다.
iHuman: Context를 참고하여 Question에 한국어로 답해주세요.
Context: {context}
Question: {question}
Assistant:\n
"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template,
)

llm_chain = LLMChain(prompt=prompt, llm=llm, output_parser=StrOutputParser())

# SQLite 데이터베이스에서 데이터 가져오기
def fetch_data_from_sqlite(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT id, doc FROM docs")  # 예시로 id와 텍스트 가져오기
    data = cursor.fetchall()
    conn.close()
    return data

embedder = SentenceTransformer("jhgan/ko-sroberta-multitask")
embedding_model = SentenceTransformerEmbeddings(model_name = "jhgan/ko-sroberta-multitask")

def embed_data(data):
    texts = [record[1] for record in data]
    embeddings = embedder.encode(texts, convert_to_tensor=True)
    return embeddings


def create_faiss_index(embeddings):
    d = embeddings.shape[1]  # 벡터 차원
    index = faiss.IndexHNSWFlat(d, 32)  # HNSW 인덱스, efConstruction=32
    index.hnsw.efSearch = 64  # 검색 성능 조정
    index.add(embeddings.cpu().numpy())  # 임베딩만 추가 (ID 없이 추가)
    return index


def search_similar_docs(query, index, model, data, top_k=5):
    # 검색 쿼리를 임베딩 벡터로 변환하고 CPU로 이동
    query_embedding = model.encode(query, convert_to_tensor=True).cpu().numpy()
    
    # query_embedding을 2차원 배열로 변환하여 검색
    D, I = index.search(np.array([query_embedding]), k=top_k)  # top_k개의 유사한 항목 반환

    # 검색 결과로부터 데이터 ID 추출
    results = [data[idx] for idx in I[0]]  # 상위 top_k의 인덱스에 해당하는 데이터
    return results

data = fetch_data_from_sqlite("docs.db")
embeddings = embed_data(data)

# FAISS 인덱스 생성 및 로컬에 저장
faiss_index = create_faiss_index(embeddings)
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


faiss.write_index(faiss_index, "deposit_index.idx")

# MultiQueryRetriever 생성
retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=vector_store.as_retriever(),  # 기존 vectorstore를 사용
    llm=llm_chain  # LLMChain을 통해 LLM을 연결
)


def retrieve_indices(query, retriever, top_k=5):
    results = retriever_from_llm.get_relevant_documents(query)  #i MultiQueryRetriever를 통해 검색
    context = "\n".join([doc.page_content for doc in results])
    
    return context

# 여러 쿼리 검색 예시
query = "최고 금리가 있는 예금"
similar_indices = retrieve_indices(query, retriever_from_llm)
print(f"Query: {query}, Similar Indices: {similar_indices}")

