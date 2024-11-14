from vllm import LLM, SamplingParams

# EEVE 모델 로드
model_path = '/home/sslab/LLM/Models/EEVE-Korean-Instruct-10.8B-v1.0'
llm = LLM(model=model_path, max_model_len=3200)

# 질문, 문맥, 참고 답변
question = "창원특례시에서 추진하고 있는 출산장려 정책의 출산장려금에서 첫째와 둘째 중 지급 금액이 더 큰 것은 무엇인가?"
context = "창원특례시(시장 허성무)는 갈수록 심화되는 저출산·고령화 문제에 적극 대응하고 인구유입 환경 조성을 위해 인구청년 정책에 6,734억원을 투입한다고 22일 밝혔다.\n시는 시민 생애 전반 기본권 보장을 통한 삶의 질 제고를 위해 출산·보육, 청년·일자리, 주거, 노후, 인구대응 등 6개분야 246개사업 인구정책 실행계획을 수립하고 이를 추진한다.\n▲출산장려 정책은 난임부부 시술비 지원 횟수가 최대 21회까지, 시술에 따라 1회당 최대 110만원까지 지원금이 확대된다. 정부 출산장려금인 '첫만남이용권(200만원)과 시에서 추진하고 있는 출산장려금(첫째50만원, 둘째200만원)도 계속해서 동시 지급 한다. 이와 함께 임산부 대상 공영주차장 주차요금 감면 정책도 하반기 시행한다.\n▲책임보육으로 아이키우기 좋은 도시환경을 만들어 나간다."
reference_answer = "계산식: null, 계산 타입: 양자/다자비교, 답: 둘째"

# 모델 입력 구성
input_text = (
    f"질문, 문맥에 대한 참고 답안을 고려하여 답안을 100단어 안으로 갼락히 응답해주세요."
    f"질문: {question}\n"
    f"문맥: {context}\n"
    f"참고 답안: {reference_answer}\n"
    f"답변:"
)

# SamplingParams 설정
sampling_params = SamplingParams(
    temperature=0.7, max_tokens=100, top_p=0.9
)

# 답변 생성
output = llm.generate(input_text, sampling_params)
generated_answer = output[0].outputs[0].text

print(f"생성된 답변: {generated_answer}")

