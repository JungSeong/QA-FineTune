simple_question_prompt_kr = Prompt(
    name="simple_question_generation_kr",
    instruction="""주어진 문맥에 기반하여 하나의 간단하고 직관적인 질문을 생성하세요.
질문은 문맥의 사실적인 내용을 직접적으로 묻는 것이어야 하며, 복잡한 추론이 필요 없어야 합니다.
다른 텍스트는 추가하지 말고, 아래 JSON 형식으로만 출력하세요:

{
  "question": "생성된 질문"
}""",
    examples=[
        {
            "context": "파리는 프랑스의 수도이며 에펠탑으로 유명하다.",
            "output": {"question": "프랑스의 수도는 어디인가요?"}
        },
        {
            "context": "광합성은 식물에서 빛 에너지를 화학 에너지로 변환한다.",
            "output": {"question": "광합성은 식물에서 어떤 에너지를 변환하나요?"}
        },
        {
            "context": "물은 100도에서 끓는다.",
            "output": {"question": "물이 끓는 온도는 몇 도인가요?"}
        },
    ],
    input_keys=["context"],
    output_key="output",  # 또는 "question" — generator에서 에러 나면 "output"으로 통일
    output_type="json",
    language="korean",
)

reasoning_question_prompt_kr = Prompt(
    name="reasoning_question_kr",
    instruction="""주어진 질문을 더 복잡하게 만들어, 제공된 문맥을 기반으로 한 다단계(multi-hop) 추론 질문으로 재작성하세요.
답변하려면 문맥의 정보를 여러 단계로 논리적으로 연결하거나 추론해야 합니다.
재작성 규칙:
1. 재작성된 질문은 문맥에 있는 정보만으로 완전히 답변 가능해야 함.
2. 질문 길이는 15단어 이하로 유지. 가능한 한 약어 사용.
3. 질문은 명확하고 모호함이 없어야 함.
4. '제공된 문맥에 기반하여', '문맥에 따르면' 같은 표현은 질문에 절대 포함 금지.""",
    examples=[
        {
            "question": "프랑스의 수도는 어디인가요?",
            "context": "프랑스는 서유럽에 위치한 나라로, 파리, 리옹, 마르세유 등의 도시가 있다. 파리는 에펠탑과 루브르 박물관 같은 문화 랜드마크로 유명하며 행정 중심지이기도 하다.",
            "output": "에펠탑과 행정 중심지를 연결하면, 어느 도시가 둘 다 해당하나요?",
        },
        {
            "question": "파이썬에서 append() 메서드는 무엇을 하나요?",
            "context": "파이썬에서 리스트는 단일 변수에 여러 항목을 저장하는 데 사용된다. append() 메서드는 리스트 끝에 단일 항목을 추가한다.",
            "output": "리스트가 변수의 가변 컬렉션을 나타낸다면, 한 항목을 확장하는 메서드는 무엇인가?",
        },
    ],
    input_keys=["question", "context"],
    output_key="output",
    output_type="str",
    language="korean",  # ← 추가 추천
)

multi_context_question_prompt_kr = Prompt(
    name="multi_context_question_kr",
    instruction="""주어진 질문을 재작성하여, context1과 context2 둘 다의 정보를 종합해야 답변 가능한 복잡한 질문으로 만드세요.
규칙:
1. 재작성 질문은 너무 길지 않게. 약어 적극 사용.
2. 질문은 합리적이고 사람이 이해하기 쉬워야 함.
3. 질문은 context1 + context2 정보만으로 완전히 답변 가능해야 함.
4. 두 문맥을 모두 읽고 이해한 후, 양쪽 통찰이 필요한 방향으로 재작성.
5. '제공된 문맥에 기반하여' 같은 표현 금지.""",
    examples=[
        {
            "question": "식물이 초록색인 이유는 무엇인가요?",
            "context1": "클로로필은 식물에 녹색을 주는 색소이며 광합성을 돕는다.",
            "context2": "광합성은 주로 엽록체가 집중된 잎에서 일어난다.",
            "output": "식물의 녹색을 만드는 색소는 어느 구조에서 에너지 생산을 돕나요?",
        },
        # 두 번째 예시도 비슷하게 한국어로 번역
    ],
    input_keys=["question", "context1", "context2"],
    output_key="output",
    output_type="str",
    language="korean",
)