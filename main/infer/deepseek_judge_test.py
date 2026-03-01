from deepeval.models import DeepSeekModel
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams, LLMTestCase
from dotenv import load_dotenv
import os

load_dotenv()

# 1. 나만의 커스텀 채점 기준표(Rubric) 작성 (원하는 대로 막 쓰시면 됩니다!)
custom_rubric = """
    이용자의 질문(Question)과 사서의 답변(Answer) 사이의 문맥적 흐름과 자연스러움을 1~5점으로 평가하세요.
    - 5점: 문맥이 매우 자연스럽고, 실제 도서관 사서처럼 대화가 부드럽게 이어짐.
    - 4점: 대체로 자연스러우나, 단어 선택이 약간 기계적인 느낌이 있음.
    - 3점: 문맥은 이어지나, 말투가 어색하거나 딱딱해서 챗봇 티가 많이 남.
    - 2점: 질문의 의도를 약간 벗어났거나 문맥이 뚝 끊기는 느낌이 듦.
    - 1점: 완전히 엉뚱한 답변(동문서답)이거나 문맥이 전혀 맞지 않음.
"""

api_key = os.getenv("DEEPSEEK_API_KEY")
deepseek_judge = DeepSeekModel(api_key=api_key, model="deepseek-chat")
print(deepseek_judge)

# 2. 커스텀 지표(Metric) 객체 생성
context_metric = GEval(
    name="Context_Naturalness", # 지표의 멋진 이름
    criteria=custom_rubric,     # 방금 만든 채점 기준표 주입!
    # 평가할 때 뭘 보고 평가할지 지정 (질문=INPUT, 답변=ACTUAL_OUTPUT)
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model=deepseek_judge,         # 우리의 똑똑한 심판관 Gemini Pro 연결
    strict_mode=True            # 점수를 무조건 기준표 안에서만 주도록 강제
)

# 3. 테스트할 데이터 1건 세팅
test_case = LLMTestCase(
    input="연체 중인데 예약한 책 대출 되나요?", 
    actual_output="음... 연체 중이면 안 될걸요? 아마 그럴 겁니다." 
)

# 4. 채점 실행!
print("DeepSeeek 심판관이 문맥을 채점 중입니다...\n")
context_metric.measure(test_case)

# 5. 결과 확인
print(f"✅ 최종 점수: {context_metric.score}점 / 5점") # DeepEval이 알아서 0~1 사이로 스케일링하거나 그대로 줍니다.
print(f"📝 채점 사유: {context_metric.reason}")