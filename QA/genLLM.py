from datasets import load_dataset
from transformers import AutoTokenizer

# 데이터셋 로드
dataset = load_dataset("json", data_files="enhance_ds.json")
# 훈련/검증 데이터로 분리 (80% 훈련, 20% 검증)
split_dataset = dataset["train"].train_test_split(test_size=0.05)
# 토크나이저 로드
model_name = "/home/sslab/LLM/Models/EEVE-Korean-Instruct-10.8B-v1.0"  # 모델 이름
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 데이터 전처리 함수
def preprocess_function(examples):
    inputs = [f"Context: {context} Question: {question}" for context, question in zip(examples["context"], examples["question"])]
    targets = examples["answer"]
    
    # 토큰화
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length").input_ids

    # 라벨에 패딩을 -100으로 설정 (loss 계산 제외)
    labels_with_ignore_index = [
        [(label if label != tokenizer.pad_token_id else -100) for label in label_seq]
        for label_seq in labels
    ]
    model_inputs["labels"] = labels_with_ignore_index
    return model_inputs

# 데이터셋 토큰화
tokenized_datasets = split_dataset.map(preprocess_function, batched=True)

from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

# 모델 로드
model = AutoModelForCausalLM.from_pretrained(model_name)

# LoRA 설정
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,  # QA 작업에 적합
    inference_mode=False,
    r=32,  # LoRA 랭크 (16~32 추천)
    lora_alpha=32,  # 학습률 스케일링
    lora_dropout=0.1,  # 드롭아웃 확률
)

# LoRA 적용
model = get_peft_model(model, lora_config)

from transformers import TrainingArguments, Trainer

# 학습 설정
training_args = TrainingArguments(
    output_dir="./results",  # 결과 저장 폴더
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    num_train_epochs=4,  # 에폭 수
    save_strategy="epoch",
    save_total_limit=2,  # 저장 개수 제한
    logging_dir="./logs",  # 로그 저장 폴더
    logging_steps=500,
    fp16=True  # 혼합 정밀도 활성화
)

# Trainer 설정
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    tokenizer=tokenizer,
)

trainer.train()

model.save_pretrained("./lora_finetuned_model")
tokenizer.save_pretrained("./lora_finetuned_model")



