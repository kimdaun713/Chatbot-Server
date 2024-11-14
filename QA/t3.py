import json

# 기존 데이터를 로드
with open("economy_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# QA 데이터셋 개선
enhanced_qa_dataset = []

# answer 파싱 함수
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

# 데이터 변환
for item in data["data"]:
    context = item["passage"]
    qa_pairs = item.get("qa_pairs", [])
    for qa in qa_pairs:
        question = qa["question"]
        answer = qa["answer"]
        
        # answer를 파싱
        parsed_answer = parse_answer(answer)
        print(question)
        print(context)
        print(parsed_answer)
        # 새로운 데이터 추가
        enhanced_qa_dataset.append({
            "문맥": context,
            "질문": question,
            "파싱된 답변": parsed_answer
        })

