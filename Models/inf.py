import torch
from vllm import LLM, SamplingParams

import torch

print("PyTorch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("Is CUDA available?", torch.cuda.is_available())

model_path = '/home/sslab/LLM/Models/EEVE-Korean-Instruct-10.8B-v1.0'

llm = LLM(model=model_path, max_model_len=2400)

prompt = """
당신은 인공지능 어시스트입니다.  Question에 계산 과정을 포함하지 않고 간결하게 답만 한국어로 답해주세요.
iHuman:  Question에 계산 과정을 포함하지 않고 간결하게 답만 한국어로 답해주세요.
Question:  "12000000원을 연이자 2%로 전세 자금 대출로 받았다면 달마다 내야할 이자가 얼마야?"
Assistant:\n
"""
sampling_params = SamplingParams(max_tokens=512)
outputs = llm.generate(prompt, sampling_params)
print(prompt)
for output in outputs:
    outputs = llm.generate(prompt, sampling_params)
    print(output.outputs[0].text, "\n\n", sep="")

