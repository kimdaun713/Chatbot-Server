from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from vllm import LLM, SamplingParams


app = FastAPI()

model_path = '/home/sslab/LLM/Models/EEVE-Korean-Instruct-10.8B-v1.0'
llm = LLM(model=model_path, max_model_len=2704)


class PromptRequest(BaseModel):
    prompt: str
    temperature: float = 0.3
    top_p: float = 0.2

@app.post("/generate")
async def generate_response(request: PromptRequest):
    try: 
        sampling_params = SamplingParams(max_tokens=256, temperature =request.temperature, top_p=request.top_p)
        output = llm.generate(request.prompt, sampling_params)
        generated_text = output[0].outputs[0].text
        return generated_text
    except Exception as e:
        raise HTTPException(status_code=500, detail = str(e))

