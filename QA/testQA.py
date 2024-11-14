import json 
from vllm import LLM, SamplingParams

# EEVE 모델 로드
model_path = '/home/sslab/LLM/Models/EEVE-Korean-Instruct-10.8B-v1.0'
llm = LLM(model=model_path, max_model_len=3000,preemption_mode='swap',swap_space=16, enable_prefix_caching = True)
sampling_params = SamplingParams(
    temperature=0.7, max_tokens=300, top_p=0.9
)
count = 0
# 답변 생성 함수
def generate_enhanced_answer(pt_buf):
    try:
        with open('enhance_ds.json', 'r', encoding='utf-8') as f:
            d = json.load(f)
    except FileNotFoundError:
        d = []

    """
    문맥, 질문, 기존 Answer를 기반으로 구체적인 답변을 생성합니다.
    """
    prompts = []
    for item in pt_buf:
        context = item["context"]
        question = item["question"]
        existing_answer = item["answer"]
    # 프롬프트 생성
        prompt = (
            f"질문, 문맥에 대한 참고 답안을 고려하여 이유와 답안을 200단어 안으로 갼락히 응답해주세요."
            f"질문: {question}\n"
            f"문맥: {context}\n"
            f"참고 답안: {existing_answer}\n"
            f"답변:"
        )

        prompts.append(prompt)
        #global count
        #count = count +1
    outputs = llm.generate(prompts, sampling_params)
    for result, item in zip(outputs, pt_buf):
        context= item["context"]
        question = item["question"]
        gen_answer = result.outputs[0].text
        print(gen_answer)
        d.append({
                "context": context,
                "question": question,
                "answer": gen_answer
            })
    with open('enhance_ds.json', 'w', encoding='utf-8') as f:
        json.dump(d, f, ensure_ascii=False, indent=4)

input_file = "economy_data.json"  # 기존 데이터 파일 경로

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)


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
cnt = 1
size = 1000
buf_ds = []
for item in data["data"]:
    if cnt %size == 0:
        generate_enhanced_answer(buf_ds)
        cnt = 1
        buf_ds = []
    else:
        cnt = cnt + 1
        context = item["passage"]
        qa_pairs = item.get("qa_pairs", [])
        for qa in qa_pairs:
            question = qa["question"]
            answer = qa["answer"]

            # answer를 파싱
            existing_answer = parse_answer(answer)
        # 새로운 데이터 추가
            buf_ds.append({
                "context" : context,
                "question": question,
                "answer": existing_answer
                })
            


#print(f"Enhanced QA Dataset has been saved to {output_file}")
#print(f"data count : {count}")
