import json 
from vllm import LLM, SamplingParams

# EEVE 모델 로드
model_path = '/home/sslab/LLM/Models/EEVE-Korean-Instruct-10.8B-v1.0'
llm = LLM(model=model_path, max_model_len=3000)
sampling_params = SamplingParams(
    temperature=0.7, max_tokens=300, top_p=0.9
)
count = 0
# 답변 생성 함수
def generate_enhanced_answer(context, question, existing_answer):
    """
    문맥, 질문, 기존 Answer를 기반으로 구체적인 답변을 생성합니다.
    """
    # 프롬프트 생성
    prompt = (
        f"질문, 문맥에 대한 참고 답안을 고려하여 이유와 답안을 200단어 안으로 갼락히 응답해주세요."
        f"질문: {question}\n"
        f"문맥: {context}\n"
        f"참고 답안: {existing_answer}\n"
        f"답변:"
    )
    global count
    count = count +1
    output = llm.generate(prompt, sampling_params)
    generated_answer = output[0].outputs[0].text
    return generated_answer

input_file = "economy_data.json"  # 기존 데이터 파일 경로
output_file = "enhanced_qa_dataset.jsonl"  # 개선된 데이터셋 저장 경로

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# QA 데이터셋 개선
enhanced_qa_dataset = []

def parse_answer(answer):
    if answer.get("spans") is not None:
        # spans 처리
        spans = answer["spans"][0]  # 첫 번째 span만 처리
        return {
            "연산식": spans.get("calculation", "N/A"),
            "연산 타입": spans.get("calculation_type", "N/A"),
            "답안": spans.get("text", "N/A"),
        }
    elif answer.get("number") is not None:
        # number 처리
        number = answer["number"]
        return {
            "연산식": number.get("calculation", "N/A"),
            "연산 타입": number.get("calculation_type", "N/A"),
            "답안": number.get("number", "N/A"),
            "한국어 번역": number.get("transcription", "N/A"),
            "단위": number.get("unit", "N/A"),
        }
    elif answer.get("date") is not None:
        # date 처리
        date = answer["date"]
        return {
            "연산 타입": date.get("calculation_type", "N/A"),
            "답안": {
                "연도": date.get("year", "N/A"),
                "월": date.get("month", "N/A"),
                "일": date.get("day", "N/A"),
            },
        }
    else:
        # 모든 필드가 null인 경우
        return {"연산 타입": "N/A", "답안": "N/A"}

for item in data["data"]:
    context = item["passage"]
    qa_pairs = item.get("qa_pairs", [])
    for qa in qa_pairs:
        question = qa["question"]
        answer = qa["answer"]

        # answer를 파싱
        existing_answer = parse_answer(answer)
        detailed_answer = generate_enhanced_answer(context, question, existing_answer)
        print(detailed_answer)
        # 새로운 데이터 추가
        enhanced_qa_dataset.append({
            "context": context,
            "question": question,
            "answer": detailed_answer
        })

# 결과 저장
with open(output_file, "w", encoding="utf-8") as f:
    for entry in enhanced_qa_dataset:
        json.dump(entry, f, ensure_ascii=False)
        f.write("\n")

print(f"Enhanced QA Dataset has been saved to {output_file}")
print(f"data count : {count}")
