from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from langchain.llms import HuggingFacePipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain.chains import RetrievalQA
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers.multi_query import MultiQueryRetriever
import logging
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = FastAPI()

loader = WebBaseLoader("https://n.news.naver.com/mnews/article/003/0012317114?sid=105")
data = loader.load()


# Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=0)
splits = text_splitter.split_documents(data)  

#model_path = '/home/sslab/LLM/Models/EEVE-Korean-Instruct-10.8B-v1.0'
#llm = LLM(model=model_path, max_model_len=1680)
#tokenizer = AutoTokenizer.from_pretrained(model_path)
#model = AutoModelForCausalLM.from_pretrained(model_path)
#hf_pipeline = pipeline("text-generation",model=model,tokenizer=tokenizer,max_new_tokens=1024,temperature=0.3)

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
A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
Human: Context를 읽고 Question에 한국어로 답하세요.
Context: {context}
Question: {question}
Assistant:\n
"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template,
)

llm_chain = prompt | llm | StrOutputParser()

#llm = HuggingFacePipeline(pipeline=hf_pipeline)

# VectorDB
#model_name = "jhgan/ko-sbert-nli"
#encode_kwargs = {'normalize_embeddings': True}
#ko_embedding = HuggingFaceEmbeddings(model_name=model_name, encode_kwargs=encode_kwargs)

db = FAISS.from_documents(splits, HuggingFaceEmbeddings(model_name='BAAI/bge-large-en-v1.5'))

retriever = db.as_retriever(search_type="similarity", search_kwargs={'k': 5})

#retriever_from_llm = MultiQueryRetriever.from_llm(retriever = retriever, llm=llm)

#qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

#vectordb = Chroma.from_documents(documents=splits, embedding=ko_embedding)


class PromptRequest(BaseModel):
    prompt: str

@app.post("/generate")
async def generate_response(request: PromptRequest):
    try: 
        #query_embedding = Open
        #full_content = ""

        logger.info("start retriever")
        #docs = retriever_from_llm.get_relevant_documents(query=request.prompt)
        #full_content = "\n".join(doc.page_content for doc in docs)
        #logger.info(f"Full content retrieved: {full_content}")

        #rag_prompt = f"Context: \n{full_content}\n\nQuestion: {request.prompt}"
        #logger.info(f"RAG prompt: {rag_prompt}")
        #combined_context = "\n".join(" ".join(doc.page_content) if isinstance(doc.page_content, list) else doc.page_content for doc in docs) 
        
#        prompt_template = PromptTemplate(input_variables=["context", "question"], template="Context:\n{context}\n\nQuestion:{question}")

        rag_chain = ({"context": retriever, "question": RunnablePassthrough()} | llm_chain)
        #inputs = tokenizer(request.prompt, return_tensors="pt")  # 입력을 텐서로 변환
        #output = hf_pipeline.model.generate(**inputs,max_length=2048)
        result = llm_chain.invoke({"context":"", "question": request.prompt})
        #response_text = tokenizer.decode(output[0], skip_special_tokens=True)
        return {"response": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail = str(e))


if __name__ == "__main__":
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
