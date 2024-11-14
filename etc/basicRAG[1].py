from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from langchain.llms import HuggingFacePipeline
from langchain.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers.multi_query import MultiQueryRetriever
import logging
import tempfile
import os
import io
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)




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

prompt = "4000원은 천원으로 변환하면 얼마야?"

llm_chain = prompt | llm | StrOutputParser()




query = "예금 상품명 KDBdream 기업 Account | 상품의 특징 고객 상황에 맞는 이 자지급방식 선택가능한 상품으로 기업의 결제성 자금의 일시예치, 일시적 여유자금의 단기적 운용이 가능, 가입 대상 기업,개인사업자 | 가입 목적 / 상품 목적  입출금자유상품 | 가입 채널/가입 경로  영업점 | 최고  금리 1.85% 10억원이상 | 가입 기간 제한없음."
        
db = FAISS.from_documents(chunkedData, HuggingFaceEmbeddings(model_name ="jinaai/jina-embeddings-v2-base-en"))
        
retriever = db.as_retriever(search_type="similarity", search_kwargs={'k': 5})
        #logger.info("start retriever")
relevant_docs = retriever.get_relevant_documents(prompt)
        #for i, doc in enumerate(relevant_docs, 1):
        #    logger.info(f"Document {i}: {doc.page_content}")
context = "\n".join([doc.page_content for doc in relevant_docs])
        
result = llm_chain.invoke({"context": context, "question": prompt})

