import os
import gc
import torch
import sys
import anyio
import re
import json
import textwrap
import uuid
import asyncio
from mcp.server.fastmcp import FastMCP
from typing import List, Dict
from vllm import LLM, SamplingParams, AsyncLLMEngine, AsyncEngineArgs
from transformers import AutoTokenizer

# 1. 경로 및 로거 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from logger_config import get_infer_logger
logger = get_infer_logger()

# 2. 환경 변수 및 전역 변수
MODEL_BASE_DIR = os.getenv("MODEL_BASE_DIR", "/models")
MCP_PORT = int(os.getenv("MCP_PORT", "4200"))
mcp = FastMCP("vLLM-Server", host="0.0.0.0", port=MCP_PORT)

llm = None
tokenizer = None
current_model_id = None

# --- [Helper Functions] ---
def resolve_model_path(model_id: str) -> str:
    """로컬 경로 우선 확인 후 모델 경로 반환"""
    local_path = os.path.join(MODEL_BASE_DIR, model_id)
    if os.path.exists(local_path):
        return local_path
    return model_id

def get_stop_token_ids():
    """모델별 종료 토큰 ID 추출"""
    global tokenizer
    if not tokenizer: return None
    stop_ids = []
    if tokenizer.eos_token_id: stop_ids.append(int(tokenizer.eos_token_id))
    for tok in ["[|endofturn|]", "[|assistant|]", "[|user|]", "[|system|]"]:
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid != tokenizer.unk_token_id: stop_ids.append(int(tid))
    return list(set(stop_ids))

# --- [MCP Tools] ---

@mcp.tool()
def unload_model():
    """VRAM 해제 및 모델 언로드"""
    global llm, tokenizer, current_model_id
    if llm is not None:
        del llm
        if tokenizer is not None: del tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        llm = None
        tokenizer = None
        current_model_id = None
        logger.info("🗑️ 모델 언로드 및 GPU 메모리 정리 완료")
        return "✅ 모델 언로드 성공"
    return "ℹ️ 로드된 모델 없음"

@mcp.tool()
async def switch_model(model_id: str, config: dict = None) -> str:
    global llm, tokenizer, current_model_id
    target_path = resolve_model_path(model_id)
    
    if current_model_id == target_path and llm is not None:
        return f"현재 이미 {model_id} 로드 중"

    unload_model() 
    logger.info(f"🌟 새롭게 load할 모델 ID || {model_id}")
    
    config = config or {}
    model_id_lower = model_id.lower()

    if "exaone" in model_id_lower:
        logger.info("🧠 Exaone 모델 감지: bfloat16 최적화 모드로 로드합니다.")
        engine_args = AsyncEngineArgs(
            model=target_path,
            dtype="bfloat16",
            trust_remote_code=True,
            gpu_memory_utilization=config.get("gpu_memory_utilization", 0.8),
            tensor_parallel_size=config.get("tensor_parallel_size", 1),
            max_model_len=config.get("max_model_len", 8192)
        )
    elif "qwen" in model_id_lower :
       engine_args = AsyncEngineArgs(
            model=target_path,
            dtype="float16",
            quantization="awq",
            trust_remote_code=True,
            gpu_memory_utilization=config.get("gpu_memory_utilization", 0.8),
            tensor_parallel_size=config.get("tensor_parallel_size", 1),
            max_model_len=config.get("max_model_len", 8192)
        ) 
    else: # A.X-Light 또는 기타 모델
        logger.info(f"🧠 {model_id} (Light/Default) 모델 감지: 기본 모드로 로드합니다.")
        engine_args = AsyncEngineArgs(
            model=target_path,
            dtype="auto",
            trust_remote_code=True,
            gpu_memory_utilization=config.get("gpu_memory_utilization", 0.8),
            tensor_parallel_size=config.get("tensor_parallel_size", 1),
            max_model_len=config.get("max_model_len", 8192),
        )

    logger.info(f"🔧 loaded engine_args || {engine_args}")

    try:
        logger.info(f"🔄 Async 모델 로드 시작: {model_id}")
        
        # AsyncLLMEngine은 자체적으로 비동기 초기화를 지원하거나 빠르게 반환됩니다.
        llm = AsyncLLMEngine.from_engine_args(engine_args)
        
        tokenizer = AutoTokenizer.from_pretrained(target_path, trust_remote_code=True)
        current_model_id = target_path
        
        logger.info(f"✅ 비동기 엔진 로드 완료: {model_id}")
        return f"✅ {model_id} 로드 성공"
    except Exception as e:
        logger.error(f"❌ 로드 실패: {e}")
        return f"❌ 로드 실패: {str(e)}"

@mcp.tool()
async def generate_text(
    messages: List[Dict],
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_k: int = 20
) -> str:
    """
    주어진 메시지 리스트를 바탕으로 텍스트를 생성합니다. (비동기 스레드 지원)
    """
    global llm, tokenizer
    
    if llm is None or tokenizer is None:
        logger.error("❌ 모델이 로드되지 않은 상태에서 generate_text 호출됨")
        return "❌ Error: Model is not loaded. Please call switch_model first."

    try:
        # 1. 메시지를 모델 전용 템플릿으로 변환 (Prompt 조립)
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # 2. 샘플링 파라미터 설정
        # 32B 모델의 경우 정지 토큰(stop_token_ids)을 확실히 지정해주는 것이 좋습니다.
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=42,
            stop_token_ids=get_stop_token_ids() # 기존에 정의한 헬퍼 함수 사용
        )

        request_id = str(uuid.uuid4())
        results_generator = llm.generate(prompt, sampling_params, request_id)
        
        final_output = None
        async for request_output in results_generator:
            final_output = request_output
            
        if final_output is None: return "❌ 생성 실패"

        generated_text = final_output.outputs[0].text
        logger.info(f"✨ 생성 완료 (ID: {request_id})")
        return generated_text

    except Exception as e:
        logger.exception(f"❌ 텍스트 생성 중 오류 발생: {e}")
        return f"❌ Error: {str(e)}"

@mcp.tool()
async def evaluate_batch(
    data_list: List[Dict],
    criteria: str = "질문과 답변이 논리적으로 일관되었으며, 질문과 라벨이 서로 올바르게 논리적으로 연관되어 있는가?"
) -> List[Dict]:
    """CoT Few-shot 기반 배치 채점 툴"""
    global llm, tokenizer
    logger.info(f"🚨 [Evaluate] 요청 수신: {len(data_list)}건")

    if not llm or not tokenizer:
        return [{"error": "모델 미로드 상태"}]

    # 1. CoT 및 배점 가중치 주입 (Few-shot)
    FEW_SHOT = textwrap.dedent("""\
        [예시 1 - 높은 점수 케이스]
        기준: {criteria}
        질문: "저는 대학생입니다. 회원증을 대리 발급 받을 수 있나요?"
        상황: "회원증 대리발급은 만14세 미만 아동, 만65세 이상 어르신, 장애인, 임산부만 가능하며 아래 구비서류를 지참하여 대리인이 방문 시 대리발급이 가능합니다.
            · 대상 : 만14세 미만 아동, 만65세 이상 어르신, 장애인, 임산부
            · 구비서류
              - 공통 : ①위임자 신분증, ②피위임자(대리인) 신분증
              - 장애인 : 장애인 복지카드 또는 장애인 증명서
              - 임신부 : 산모수첩 / 산모 : 주민등록등본(출산 후 12개월까지)
            · 방법 : 홈페이지 회원가입 후 위 항목의 해당하는 구비서류를 지참하여 피위임자(대리인)이 도서관 방문
            "
        답변: "죄송합니다. 회원증 대리발급은 만14세 미만 아동, 만65세 이상 어르신, 장애인, 임산부만 가능하며 아래 구비서류를 지참하여 대리인이 방문 시 대리발급이 가능합니다."
        라벨: "no"

        [추론 과정]
        1. 기준 이해: {criteria}
        2. label 유형: 'no' → 부정적 가부 답변을 기대
        3. 상황 일관성: 질문에서 본인이 대학생임을 언급하였음. 상황에서는 회원증 대리 발급은 만14세 미만 아동, 만 65세 이상 어르신, 장애인, 임산부만 가능하다고 하였으며, 따라서 대학생은 이에 해당하지 않음을 알 수 있음. '장애가 있는 대학생'일 가능성도 존재하나, 답변과 상황이 논리적으로 상당히 연관되어 있음 -> 우수 (4/5점)
        4. 논리 일관성: 답변은 질문자의 상황(대학생)을 고려하여 대리발급 불가능을 명확히 전달하면서, 상황의 세부 사항(대상자, 구비서류, 방법)을 그대로 인용하여 설명함. 불필요한 추가 정보 없이 핵심에 집중되어 논리적 흐름이 자연스럽고, 부정적인 맥락을 '죄송합니다'로 부드럽게 처리함 → 우수 (3/3점)
        5. label 유사성: 라벨이 'no'인데 답변도 부정적(대리발급 불가능함을 암시)이며, 구체적인 이유와 정보를 제공하여 라벨의 부정성을 강화 → 매우 우수 (7/7점)
        → 종합적으로 거의 완벽에 가까움

        [예시 2 - 낮은 점수 케이스]
        기준: {criteria}
        질문: "상호대차로 빌린 책을 다른 도서관에 반납해도 되나요?"
        상황: "상호대차 신청이 불가한 경우  
        1. 신청이 불가한 자료의 경우  
        대출중인 도서, 예약도서, 참고도서, 비도서(DVD) 등  
        2. 신청이 제한된 상태의 경우  
        - 도서연체 또는 대출정지 상태일 경우 상호대차 이용 불가  
        ※ 상호대차 도착안내 메시지를 수신하였더라도 이용 불가  
        - 상호대차 신청 후 미대출 5권이 되면 1개월(30일)간 신청제한  
        3. 이용이 불가한 도서관의 경우  
        별꿈도서관, 작은도서관(수령만 가능) , 스마트 도서관  
        ※ 시립도서관 휴관일 상호대차 신청불가"
        답변: "네 가능합니다, 상호대차로 빌린 책도 다른 시립도서관에 반납 가능합니다. 편하게 가까운 도서관에 넣으세요!"
        라벨: "yes"

        [추론 과정]
        1. 기준 이해: {criteria}
        2. label 유형: 'yes' → 긍정적 가부 답변을 기대
        3. 상황 일관성: 상황에 따르면 상호대차는 "작은도서관(수령만 가능)", "스마트도서관" 등 특정 도서관군에서 제한이 명확하고, 특히 시립도서관의 경우 휴관일에 상호대차가 신청 불가능함. 하지만 답변은 이를 완전히 무시하고 "다른 시립도서관에 반납 가능"이라고 잘못 전달함 → 매우 낮음 (0/5점)
        4. 논리 일관성: 답변 내용이 상황의 핵심 규칙(상호대차 반납 제한)을 정면으로 부정하며, 오히려 잘못된 정보를 제공함. 사용자가 오해할 가능성이 매우 높아 논리적으로 매우 비약이 심함 → 매우 낮음 (0/3점)
        5. label 유사성: 라벨이 'yes'(가능)인데 답변 또한 긍정적('네, 가능합니다')으로 답변하고 있음. 따라서 답변과 라벨은 논리적으로 연관되었음을 볼 수 있음 → 매우 우수 (7/7점)
        → 종합적으로 기준과 어긋나서 낮은 점수 케이스나, 답변과 라벨은 논리적으로 연관되어 있어 점수가 보정됨 
    """)

    prompts = []
    request_ids = []

    for item in data_list:
        prompt_text = textwrap.dedent(f"""
            당신은 도서관 QA 데이터의 품질을 매우 엄격하고 일관되게 평가하는 전문가입니다.
            주어진 QA 쌍을 다음 **추론 과정**대로 분석한 뒤 JSON으로만 답변하세요.
            절대 다른 텍스트를 추가하지 마세요.

            {FEW_SHOT}

            [추론 과정 - 반드시 이 순서를 지켜 생각하세요]
            1. 기준 이해 및 키워드 정리: 기준은 "{criteria}"이며 특히 답변과 라벨 '{item.get('label', '')}'의 유사성을 중점 평가
            2. label 유형 확인: 'yes' → 긍정 가부, 'no' → 부정 가부, 'info' → 설명형, 'false' → 모호/오류
            3. 상황 일관성 점검 (최대 5점): 답변은 해당 상황에 정확힌 근거하였는가?
            4. 논리 일관성 점검 (최대 3점): 질문이 요구하는 형식(가부/정보)에 답변이 정확히 대응하는가?
            5. label 유사성 점검 (최대 7점): 라벨과 답변의 내용·극성·구체성이 얼마나 잘 맞는가?
            6. 종합 점수 계산: 0~15점 스케일 (위 두 항목 합산 후 보정)
            7. thought 작성: 위 1~6번 과정을 압축해서 서술 (구체적인 점수 배점 언급 권장)

            [대상 데이터]
            질문: {item.get('question', '')}
            답변: {item.get('answer', '')}
            상황: {item.get('context', '')}
            라벨: {item.get('label', '')}

            최종 출력은 **JSON 형식 한 개**만, thought는 3~7문장 정도로:
            {{
                "thought": "...",
                "score": 정수,
                "reason": "한 문장 요약 근거"
            }}
        """)

        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            tokenize=False,
            add_generation_prompt=True
        )
        prompts.append(formatted)
        request_ids.append(str(uuid.uuid4()))

    if not prompts:
        return []

    params = SamplingParams(temperature=0.1, max_tokens=1024, seed=42, stop_token_ids=get_stop_token_ids())
    generators = [
        llm.generate(prompt, params, req_id)
        for prompt, req_id in zip(prompts, request_ids)
    ]

    final_outputs = [None] * len(data_list)

    async def collect(idx: int, gen):
        last = None
        async for out in gen:
            last = out
        final_outputs[idx] = last

    await asyncio.gather(*[collect(i, g) for i, g in enumerate(generators)])

    final_results = []
    for i, output in enumerate(final_outputs):
        original_item = data_list[i]
        if output is None or not output.outputs:
            raw_text = ""
        else:
            raw_text = output.outputs[0].text.strip()

        score = 3
        parsed = {}
        try:
            # 더 나은 JSON 추출 (```json ... ```도 잡음)
            import re
            match = re.search(r'(?:```json\s*)?(\{[\s\S]*?\})(?:\s*```)?', raw_text, re.DOTALL)
            if match:
                json_str = match.group(1).strip()
                parsed = json.loads(json_str)
                score = int(parsed.get("score", 3))
        except Exception as e:
            logger.warning(f"[{i}] 파싱 실패: {e} | raw: {raw_text[:80]}...")

        final_results.append({
            "question": original_item.get('question', ''),
            "answer": original_item.get('answer', ''),
            "context": original_item.get('context', ''),
            "label": original_item.get('label', ''),
            "eval_score": score,
            "eval_reason": parsed.get("reason", "파싱 실패"),
            "eval_thought": parsed.get("thought", ""),
        })

    logger.info(f"✅ 배치 평가 완료: {len(final_results)}건")
    return final_results

@mcp.tool()
async def final_evaluate_batch(
    data_list: List[Dict],
    criteria: str = "답변에 환각(Hallucination) 현상은 없는지, 답변은 질문에 논리적으로 부합하는지, 또한 실제 사서가 말하듯 정중하게 답변하였는지 평가하라."
) -> List[Dict]:
    """CoT Few-shot 기반 배치 채점 툴 (Hallucination 킬러 버전)"""
    global llm, tokenizer

    if not llm or not tokenizer:
        return [{"error": "모델 미로드 상태"}]

    # 1. CoT 및 배점 가중치 주입 (Few-shot) - 환각(Hallucination) 검출에 초점
    FEW_SHOT = textwrap.dedent("""\
        [예시 1 - 높은 점수 케이스 (환각 없음)]
        질문: "저는 대학생입니다. 회원증을 대리 발급 받을 수 있나요?"
        상황: "회원증 대리발급은 만14세 미만 아동, 만65세 이상 어르신, 장애인, 임산부만 가능하며 아래 구비서류를 지참하여 대리인이 방문 시 대리발급이 가능합니다.
            · 대상 : 만14세 미만 아동, 만65세 이상 어르신, 장애인, 임산부
            · 구비서류
              - 공통 : ①위임자 신분증, ②피위임자(대리인) 신분증
              - 장애인 : 장애인 복지카드 또는 장애인 증명서
              - 임신부 : 산모수첩 / 산모 : 주민등록등본(출산 후 12개월까지)
            · 방법 : 홈페이지 회원가입 후 위 항목의 해당하는 구비서류를 지참하여 피위임자(대리인)이 도서관 방문
            "
        답변: "죄송합니다. 회원증 대리발급은 만14세 미만 아동, 만65세 이상 어르신, 장애인, 임산부만 가능하며 아래 구비서류를 지참하여 대리인이 방문 시 대리발급이 가능합니다."

        [추론 과정]
        1. 환각 여부 검증: 상황(Context)에 명시된 대리발급 대상자(14세 미만, 65세 이상, 장애인, 임산부)를 정확히 인용하였으며, 규정에 없는 어떠한 내용도 지어내지 않음 (환각 없음) → 완벽함 (8/8점)
        2. 논리 일관성: 질문자의 상황(대학생)에 맞춰 대리발급이 불가함을 명확하고 논리적으로 안내함 → 우수 (5/5점)
        3. 사서 페르소나: "죄송합니다"라고 정중하게, 사서처럼 친절하게 답변함 → 우수 (2/2점)

        ```json
        {
            "thought": "제공된 규정(상황)을 100% 준수하여 정보의 왜곡이나 환각(Hallucination)이 전혀 없음. 질문자의 의도를 정확히 파악하여 논리적으로 올바른 거절 안내를 하였고, 사서의 정중한 페르소나를 잘 유지함.",
            "score_faithfulness": 8,
            "score_relevancy": 5,
            "score_persona": 2,
            "total_score": 15,
            "reason": "환각 없이 완벽하게 규정에 근거한 논리적이고 정중한 답변"
        }
        ```

        [예시 2 - 낮은 점수 케이스 (치명적 환각 발생)]
        질문: "상호대차로 빌린 책을 다른 도서관에 반납해도 되나요?"
        상황: "상호대차 신청이 불가한 경우  
        1. 신청이 불가한 자료의 경우  
        대출중인 도서, 예약도서, 참고도서, 비도서(DVD) 등  
        2. 신청이 제한된 상태의 경우  
        - 도서연체 또는 대출정지 상태일 경우 상호대차 이용 불가  
        ※ 상호대차 도착안내 메시지를 수신하였더라도 이용 불가  
        - 상호대차 신청 후 미대출 5권이 되면 1개월(30일)간 신청제한  
        3. 이용이 불가한 도서관의 경우  
        별꿈도서관, 작은도서관(수령만 가능) , 스마트 도서관  
        ※ 시립도서관 휴관일 상호대차 신청불가"
        답변: "네 가능합니다, 상호대차로 빌린 책도 다른 시립도서관에 반납 가능합니다. 편하게 가까운 도서관에 넣으세요!"

        [추론 과정]
        1. 환각 여부 검증: 상황(Context)에 따르면 상호대차 제한이 명확함에도 불구하고, 이를 완전히 무시하고 "다른 시립도서관에 반납 가능"이라는 허위 정보(Hallucination)를 창작하여 제공함. 가장 치명적인 오류임 → 매우 낮음 (0/8점)
        2. 논리 일관성: 오안내를 통해 질문자에게 완전히 잘못된 결론을 유도함 → 매우 낮음 (0/5점)
        3. 사서 페르소나: 답변의 말투 자체는 긍정적이고 정중함 → 우수 (2/2점)

        ```json
        {
            "thought": "말투는 정중하나, 제공된 도서관 규정을 정면으로 위반하고 없는 규정을 지어내는 치명적인 환각(Hallucination) 오류를 범함. 이로 인해 질문자에게 잘못된 정보를 전달하였으므로 논리 일관성에서도 최하점을 부여함.",
            "score_faithfulness": 0,
            "score_relevancy": 0,
            "score_persona": 2,
            "total_score": 2,
            "reason": "말투는 정중하나 규정을 완전히 위반한 치명적 환각(Hallucination) 발생"
        }
        ```
    """)

    prompts = []
    request_ids = []

    for item in data_list:
        prompt_text = textwrap.dedent(f"""
            당신은 도서관 QA 데이터의 품질을 매우 엄격하고 일관되게 평가하는 전문가입니다.
            주어진 QA 쌍을 다음 **추론 과정**대로 분석한 뒤 JSON으로만 답변하세요.
            절대 다른 텍스트를 추가하지 마세요.

            {FEW_SHOT}

            [추론 과정 - 반드시 이 순서를 지켜 생각하세요]
            1. 환각 여부 검증 (최대 8점 - **가장 중요**): 답변이 상황(Context)에 명시된 사실만을 포함하고 있는가? 규정에 없는 내용을 덧붙이거나 왜곡(Hallucination)했다면 낮은 점수를 부여할 것.
            2. 논리 일관성 검증 (최대 5점): 질문의 의도를 정확히 파악하고 동문서답 없이 직접적인 결론(가부 및 정보)을 도출했는가?
            3. 사서 페르소나 점검 (최대 2점): 실제 사서처럼 말투가 정중하고 자연스러운가?
            4. 종합 점수 계산: 0~15점 스케일 (위 세 항목 합산)
            5. thought 작성: 특히 환각 여부를 중점적으로 지적하며 3~5문장으로 서술할 것.

            [대상 데이터]
            질문: {item.get('question', '')}
            답변: {item.get('answer', '')}
            상황: {item.get('context', '')}

            최종 출력은 **오직 JSON 형식 한 개**만 작성하세요.
            반드시 아래 키(Key)값들을 모두 포함해야 합니다:
            {{
                "thought": "환각 검증을 중심으로 한 3~5문장 추론 요약",
                "score_faithfulness": 정수 (0~8),
                "score_relevancy": 정수 (0~5),
                "score_persona": 정수 (0~2),
                "total_score": 정수 (0~15),
                "reason": "한 문장 요약 근거"
            }}
        """)

        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            tokenize=False,
            add_generation_prompt=True
        )
        prompts.append(formatted)
        request_ids.append(str(uuid.uuid4()))

    if not prompts:
        return []

    params = SamplingParams(temperature=0.1, max_tokens=1024, seed=42, stop_token_ids=get_stop_token_ids())
    generators = [
        llm.generate(prompt, params, req_id)
        for prompt, req_id in zip(prompts, request_ids)
    ]

    final_outputs = [None] * len(data_list)

    async def collect(idx: int, gen):
        last = None
        async for out in gen:
            last = out
        final_outputs[idx] = last

    await asyncio.gather(*[collect(i, g) for i, g in enumerate(generators)])

    final_results = []
    for i, output in enumerate(final_outputs):
        original_item = data_list[i]
        if output is None or not output.outputs:
            raw_text = ""
        else:
            raw_text = output.outputs[0].text.strip()

        parsed = {}
        try:
            import re
            match = re.search(r'(?:```json\s*)?(\{[\s\S]*?\})(?:\s*```)?', raw_text, re.DOTALL)
            if match:
                json_str = match.group(1).strip()
                parsed = json.loads(json_str)
        except Exception as e:
            logger.warning(f"[{i}] 파싱 실패: {e} | raw: {raw_text[:80]}...")

        # 파싱된 세부 점수 안전하게 가져오기
        s_faith = int(parsed.get("score_faithfulness", 0))
        s_rel = int(parsed.get("score_relevancy", 0))
        s_pers = int(parsed.get("score_persona", 0))
        t_score = int(parsed.get("total_score", 0))

        final_results.append({
            "question": original_item.get('question', ''),
            "answer": original_item.get('answer', ''),
            "context": original_item.get('context', ''),
            "eval_faithfulness": s_faith,    # 환각 방지 지표 (8점 만점)
            "eval_relevancy": s_rel,         # 동문서답 방지 지표 (5점 만점)
            "eval_persona": s_pers,          # 말투 지표 (2점 만점)
            "eval_score": t_score,           # 총점 (15점 만점)
            "eval_reason": parsed.get("reason", "파싱 실패"),
            "eval_thought": parsed.get("thought", ""),
        })

    logger.info(f"✅ 배치 세부 지표 평가 완료: {len(final_results)}건")
    return final_results

if __name__ == "__main__":
    logger.info(f"🚀 MCP 서버 가동 (Port: {MCP_PORT})")
    mcp.run(transport="sse")