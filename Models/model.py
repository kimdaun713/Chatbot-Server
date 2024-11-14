from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("yanolja/EEVE-Korean-Instruct-10.8B-v1.0", cache_dir ='.')
tokenizer = AutoTokenizer.from_pretrained("yanolja/EEVE-Korean-Instruct-10.8B-v1.0", cache_dir = '.')

prompt_template = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\nHuman: {prompt}\nAssistant:\n"
text = '6천원을 백원 단위로 변환하면 어떻게 돼?'
model_inputs = tokenizer(prompt_template.format(prompt=text), return_tensors='pt')

outputs = model.generate(**model_inputs, max_new_tokens=256)
output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
print(output_text)
