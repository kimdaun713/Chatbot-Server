import json 
from vllm import LLM, SamplingParams

try:
    with open('enhance_ds.json', 'r', encoding='utf-8') as f:
            d = json.load(f)
except FileNotFoundError:
        d = []

print(len(d)) 


#print(f"Enhanced QA Dataset has been saved to {output_file}")
#print(f"data count : {count}")
