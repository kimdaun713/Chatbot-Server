from fastapi import FastAPI, HTTPException, UploadFile, File, Form
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
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import logging
import tempfile
import os
import io
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = FastAPI()


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
당신은 인공지능 어시스트입니다. The assistant gives helpful, detailed, and polite answers to the user's questions.
Human: Context를 참고하여 Question에 한국어로 답해주세요.
Context: {context}
Question: {question}
Assistant:\n
"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template,
)

llm_chain = prompt | llm | StrOutputParser()




async def load_data(file:UploadFile):
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(await file.read())
        temp_file_path = temp_file.name

    loader = PyMuPDFLoader(temp_file_path)  
    data = loader.load()
    os.remove(temp_file_path)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(data)
    
    print(splits)
    return splits



@app.post("/generate")
async def generate_response(prompt: str=Form(...), file: UploadFile = File(...)):
    try: 
        chunked_data = await load_data(file)
        db = FAISS.from_documents(chunked_data, HuggingFaceEmbeddings(model_name ="jinaai/jina-embeddings-v2-base-en"))
        retriever_from_llm = MultiQueryRetriever.from_llm(retriever=db.as_retriever(), llm=llm)
        logger.info("start retriever")
        relevant_docs = retriever_from_llm.get_relevant_documents(query=prompt)
    
        #for i, doc in enumerate(relevant_docs, 1):
        #    logger.info(f"Document {i}: {doc.page_content}")
        context = "\n".join([doc.page_content for doc in relevant_docs])
        result = llm_chain.invoke({"context": context, "question": prompt})

        return {"response": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail = str(e))


if __name__ == "__main__":
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
