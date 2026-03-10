"""
[Exp1~4]에 대해 infer 

실행 순서:
  Baseline + CoT-LoRA + SFT-LoRA를 단일 vLLM 서버로 기동 후
  비동기로 Exp1~4 추론 → JSONL 저장 → 2_evaluate.py가 읽음

실험 설계:
  Exp1 Baseline  질문만 입력 (context 없음) → 베이스 모델 순수 실력 측정
  Exp2 RAG       질문 + 검색된 context     → retrieval 효과 측정
  Exp3 CoT-LoRA  질문 + original_title     → CoT 파인튜닝 효과 측정
  Exp4 SFT-LoRA  질문 + original_title     → SFT 효과 측정

contexts 저장 정책 (RAGAS faithfulness 평가용):
  Exp1 → original_title (추론엔 미사용, 평가 시 환각 여부 측정)
  Exp2 → retrieved docs (실제 검색 결과)
  Exp3 → original_title (추론에도 사용)
  Exp4 → original_title (추론에도 사용)

속도 측정:
  elapsed_sec : 개별 샘플 응답 시간 (비동기 큐 대기 포함 → 디버깅용)
  wall_time   : 실험 전체 시작~끝 시간 → _meta.json에 저장 (속도 비교용)
"""

import os
import sys
import json
import logging
import signal
import subprocess
import time
import asyncio
import requests
from pathlib import Path
from typing import List, Dict, Tuple
from openai import AsyncOpenAI

# ─────────────────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────────────────

TEST_DATA_PATH = "../data/golden_data/test_goldens.jsonl"
INFER_SAVE_DIR = "./infer_results"
INFER_PORT     = 8002
INFER_BASE_URL = f"http://localhost:{INFER_PORT}/v1"
# 아래 모델 파일들은 실제로 /home/vsc/LLM/model 에 위치한다
MODEL_BASELINE = "/models/Exaone-3.5-7.8B-Instruct"
MODEL_COT      = "/models/Exaone-3.5-7.8B-Instruct-SFT-CoT/final_10"
MODEL_SFT      = "/models/Exaone-3.5-7.8B-Instruct-SFT-Golden/final_10"
SEARCH_API_URL = "http://localhost:8080"
VLLM_READY_TIMEOUT  = 240
VLLM_READY_INTERVAL = 5
MAX_NEW_TOKENS      = 512
SEED = 42

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from logger_config import get_infer_logger

logger = get_infer_logger()

# ─────────────────────────────────────────────────────────
# 프롬프트
# ─────────────────────────────────────────────────────────

# Exp1 Baseline 전용 — context 없이 질문만 받는 상황
SYSTEM_BASELINE = (
    "당신은 도서관 운영에 대한 전문적인 지식을 가진 인공지능 사서입니다. "
    "사용자의 질문에 정확하고 친절하게 답하십시오. "
    "정보에 없는 내용은 함부로 추측하지 말고 정중히 확인이 어렵다고 답하세요."
)

# Exp2~4 공통 — 학습 시 generate_prompts와 동일한 포맷
SYSTEM_LIBRARY = (
    "당신은 도서관 운영에 대한 전문적인 지식을 가진 인공지능 사서입니다. "
    "제공된 [도서관 정보]를 바탕으로 사용자의 질문에 정확하고 친절하게 답하십시오. "
    "정보에 없는 내용은 함부로 추측하지 말고 정중히 확인이 어렵다고 답하세요."
)


def _build_user_content(question: str, context: str) -> str:
    """학습 시 generate_prompts와 동일한 user 메시지 포맷"""
    return (
        f"### [도서관 정보]\n{context}\n\n"
        f"### [질문]\n{question}\n\n"
        f"### [지시 사항]\n"
        f"1. 친절한 말투로 규정에 근거하여 답변할 것.\n"
        f"2. 3문단 이내로 답변할 것.\n"
        f"3. 답변 끝에 지시 사항을 반복하지 말 것."
    )


# ─────────────────────────────────────────────────────────
# 1. 데이터 로드 / 저장
# ─────────────────────────────────────────────────────────

def load_test_data(path: str) -> List[Dict]:
    records = []
    if not os.path.exists(path):
        logger.error("❌ 파일을 찾을 수 없습니다: %s", path)
        return []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning("파싱 실패 [line %d]: %s", line_no, e)
    logger.info("📁 테스트 데이터 로드 완료: %d개", len(records))
    return records


def save_infer_results(
    exp_name: str,
    records: List[Dict],
    responses: List[str],
    elapsed_list: List[float],
    contexts_list: List,
    save_dir: str,
    wall_time: float,
) -> None:
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # 1. JSONL 저장 (elapsed_sec은 디버깅용으로 유지)
    path = Path(save_dir) / f"{exp_name}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for r, resp, sec, ctxs in zip(records, responses, elapsed_list, contexts_list):
            if isinstance(ctxs, str):
                ctxs = [ctxs]
            elif not isinstance(ctxs, list):
                ctxs = []
            f.write(json.dumps({
                "question":     r["question"],
                "ground_truth": r["answer"],
                "response":     resp,
                "contexts":     ctxs,
                "elapsed_sec":  sec,
            }, ensure_ascii=False) + "\n")

    # 2. wall time 메타 저장 → 2_evaluate.py가 속도 지표로 사용
    n = len(records)
    meta = {
        "wall_time_sec":        round(wall_time, 2),
        "n_samples":            n,
        "wall_time_per_sample": round(wall_time / n, 3),
    }
    meta_path = Path(save_dir) / f"{exp_name}_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    logger.info(
        "✅ %s 저장 완료 (%d개) | wall time: %.2fs (%.3fs/sample)",
        exp_name, n, wall_time, wall_time / n,
    )

# ─────────────────────────────────────────────────────────
# 2. RemoteRetriever (RAG)
# ─────────────────────────────────────────────────────────

class RemoteRetriever:
    def __init__(self, endpoint_url: str):
        self.endpoint_url = endpoint_url
        logger.info("📡 Retriever 연결: %s", endpoint_url)

    def retrieve(self, question: str, top_k: int = 3) -> List[str]:
        try:
            resp = requests.get(
                f"{self.endpoint_url}/search",
                params={"query": question, "k": top_k},
                timeout=15,
            )
            return resp.json().get("results", []) if resp.status_code == 200 else []
        except Exception as e:
            logger.error("검색 서버 호출 실패: %s", e)
            return []


# ─────────────────────────────────────────────────────────
# 3. 비동기 추론 함수
# ─────────────────────────────────────────────────────────

async def _async_call(
    client: AsyncOpenAI,
    messages: List[Dict],
    model_id: str,
) -> Tuple[str, float]:
    """공통 비동기 LLM 호출 — (응답 텍스트, elapsed_sec) 반환"""
    start = time.perf_counter()
    try:
        res = await client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=0,
            max_tokens=MAX_NEW_TOKENS,
        )
        elapsed = time.perf_counter() - start
        content = res.choices[0].message.content or ""
        if not content.strip():
            logger.error("빈 응답 반환 (model=%s)", model_id)
        return content.strip(), elapsed
    except Exception as e:
        logger.error("❌ 호출 오류 (model=%s): %s", model_id, e)
        return "ERROR_RESPONSE", 0.0

async def warmup(client: AsyncOpenAI, model_id: str) -> None:
    """LoRA 어댑터 캐시 워밍업 — wall_time 측정 전에 호출"""
    await _async_call(client, [
        {"role": "user", "content": "안녕"}
    ], model_id)

def _get_original_title(record: Dict) -> List[str]:
    """original_title을 항상 list[str]로 반환합니다."""
    val = record.get("original_title", [])
    if isinstance(val, list):
        return val
    return [val] if val else []


async def infer_baseline_task(
    client: AsyncOpenAI,
    record: Dict,
) -> Tuple[str, float, List[str]]:
    """
    Exp1: Baseline
    - 질문만 입력 (context 없음) — 베이스 모델 순수 실력 측정
    - contexts에는 original_title 저장 (RAGAS faithfulness 평가용)
    """
    resp, sec = await _async_call(
        client,
        [
            {"role": "system", "content": SYSTEM_BASELINE},
            {"role": "user",   "content": record["question"]},
        ],
        MODEL_BASELINE,
    )
    return resp, sec, _get_original_title(record)


async def infer_rag_task(
    client: AsyncOpenAI,
    record: Dict,
    retriever: RemoteRetriever,
) -> Tuple[str, float, List[str]]:
    """
    Exp2: RAG
    - 검색된 문서를 [도서관 정보]에 넣어 학습 포맷과 동일하게 입력
    - contexts에는 retrieved docs 저장
    """
    loop = asyncio.get_event_loop()
    retrieved = retriever.retrieve(record["question"], top_k=3)

    if retrieved:
        logger.info("🔍 [RAG] 질문: %.30s | 검색 %d건", record["question"], len(retrieved))
        for i, doc in enumerate(retrieved):
            logger.info("   ㄴ 문서 %d: %.80s", i + 1, doc.replace("\n", " "))
    else:
        logger.warning("⚠️ 검색 결과 없음: %.30s", record["question"])

    context_text = "\n\n".join(retrieved) if retrieved else "관련 참고 자료가 없습니다."
    resp, sec = await _async_call(
        client,
        [
            {"role": "system", "content": SYSTEM_LIBRARY},
            {"role": "user",   "content": _build_user_content(record["question"], context_text)},
        ],
        MODEL_BASELINE,
    )
    return resp, sec, retrieved


async def infer_cot_task(
    client: AsyncOpenAI,
    record: Dict,
) -> Tuple[str, float, List[str]]:
    """
    Exp3: CoT-LoRA
    - 학습 포맷과 동일하게 [도서관 정보]에 original_title 입력
    - contexts에는 original_title 저장
    """
    original_title = _get_original_title(record)
    context_text   = "\n".join(original_title)

    resp, sec = await _async_call(
        client,
        [
            {"role": "system", "content": SYSTEM_LIBRARY},
            {"role": "user",   "content": _build_user_content(record["question"], context_text)},
        ],
        "cot-lora",
    )
    logger.info("--- CoT 추론 완료 (%.2fs)", sec)
    return resp, sec, original_title


async def infer_sft_task(
    client: AsyncOpenAI,
    record: Dict,
) -> Tuple[str, float, List[str]]:
    """
    Exp4: SFT-LoRA
    - 학습 포맷과 동일하게 [도서관 정보]에 original_title 입력
    - contexts에는 original_title 저장
    """
    original_title = _get_original_title(record)
    context_text   = "\n".join(original_title)

    resp, sec = await _async_call(
        client,
        [
            {"role": "system", "content": SYSTEM_LIBRARY},
            {"role": "user",   "content": _build_user_content(record["question"], context_text)},
        ],
        "sft-lora",
    )
    logger.info("--- SFT 추론 완료 (%.2fs)", sec)
    return resp, sec, original_title


# ─────────────────────────────────────────────────────────
# 5. 메인
# ─────────────────────────────────────────────────────────

async def main_async() -> None:
    retriever = RemoteRetriever(SEARCH_API_URL)
    try:
        records = load_test_data(TEST_DATA_PATH)
        if not records:
            logger.critical("테스트 데이터가 없습니다. 종료합니다.")
            sys.exit(1)

        client = AsyncOpenAI(base_url=INFER_BASE_URL, api_key="vllm-dummy")

        # ── Exp1: Baseline ────────────────────────────────
        logger.info("=" * 55)
        logger.info("▶ Exp1: Baseline 비동기 추론 시작 (질문만 입력)")
        t0   = time.perf_counter()
        res1 = await asyncio.gather(*[infer_baseline_task(client, r) for r in records])
        save_infer_results(
            "exp1_baseline", records,
            [x[0] for x in res1], [x[1] for x in res1], [x[2] for x in res1],
            INFER_SAVE_DIR,
            wall_time=time.perf_counter() - t0,
        )

        # ── Exp2: RAG ─────────────────────────────────────
        logger.info("=" * 55)
        logger.info("▶ Exp2: RAG 비동기 추론 시작 (retrieved context 입력)")
        await warmup(client, MODEL_BASELINE)
        t0   = time.perf_counter()
        res2 = await asyncio.gather(*[infer_rag_task(client, r, retriever) for r in records])
        save_infer_results(
            "exp2_rag", records,
            [x[0] for x in res2], [x[1] for x in res2], [x[2] for x in res2],
            INFER_SAVE_DIR,
            wall_time=time.perf_counter() - t0,
        )

        # ── Exp3: CoT-LoRA ────────────────────────────────
        logger.info("=" * 55)
        logger.info("▶ Exp3: CoT-LoRA 비동기 추론 시작 (original_title 입력)")
        await warmup(client, "cot-lora")
        t0   = time.perf_counter()
        res3 = await asyncio.gather(*[infer_cot_task(client, r) for r in records])
        save_infer_results(
            "exp3_cot", records,
            [x[0] for x in res3], [x[1] for x in res3], [x[2] for x in res3],
            INFER_SAVE_DIR,
            wall_time=time.perf_counter() - t0,
        )

        # ── Exp4: SFT-LoRA ────────────────────────────────
        logger.info("=" * 55)
        logger.info("▶ Exp4: SFT-LoRA 비동기 추론 시작 (original_title 입력)")
        await warmup(client, "sft-lora")
        t0   = time.perf_counter()
        res4 = await asyncio.gather(*[infer_sft_task(client, r) for r in records])
        save_infer_results(
            "exp4_sft", records,
            [x[0] for x in res4], [x[1] for x in res4], [x[2] for x in res4],
            INFER_SAVE_DIR,
            wall_time=time.perf_counter() - t0,
        )

    finally:
        logger.info("=" * 55)
        logger.info("✅ 모든 추론 완료 → %s", INFER_SAVE_DIR)
        logger.info("다음 단계: exaone 컨테이너 내리고 qwen 컨테이너 올린 뒤 2_evaluate.py 실행")

if __name__ == "__main__":
    asyncio.run(main_async())