"""
고양시 도서관 AI 사서 — FastAPI 미들웨어 서버
위치: QA-FineTune/main/web/server.py

역할:
  - exaone 컨테이너 vLLM OpenAI API 프록시
  - RAG 검색 서버 연동
  - 모드별 프롬프트 처리 (Baseline / CoT-LoRA / SFT-LoRA + RAG 옵션)

실행 방법 (호스트에서):
  pip install fastapi uvicorn
  uvicorn server:app --host 0.0.0.0 --port 9000

사전 조건:
  docker compose up exaone search-data postgres -d
  → exaone 컨테이너가 포트 8002에 vLLM OpenAI API를 제공해야 함
"""

import time
import logging
import requests
import os
import sys

from typing import Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

# ─────────────────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────────────────

VLLM_BASE_URL  = "http://localhost:8002/v1"
SEARCH_API_URL = "http://localhost:8080"
VLLM_API_KEY   = "vllm-dummy"
TOP_K          = 3
MAX_NEW_TOKENS = 1024

MODEL_COT = "cot-lora"
MODEL_SFT = "sft-lora"

SYSTEM_BASELINE = (
    "당신은 도서관 운영에 대한 전문적인 지식을 가진 인공지능 사서입니다. "
    "사용자의 질문에 정확하고 친절하게 답하십시오. "
    "정보에 없는 내용은 함부로 추측하지 말고 정중히 확인이 어렵다고 답하세요."
)
SYSTEM_LIBRARY = (
    "당신은 도서관 운영에 대한 전문적인 지식을 가진 인공지능 사서입니다. "
    "제공된 [도서관 정보]를 바탕으로 사용자의 질문에 정확하고 친절하게 답하십시오. "
    "정보에 없는 내용은 함부로 추측하지 말고 정중히 확인이 어렵다고 답하세요."
)

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(current_dir)
sys.path.append(parent_dir)

try:
    from logger_config import get_server_logger
    logger = get_server_logger()
except ImportError:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
    logger = logging.getLogger("server")


# ─────────────────────────────────────────────────────────
# 헬퍼
# ─────────────────────────────────────────────────────────

def _is_vllm_alive() -> bool:
    try:
        return requests.get(f"{VLLM_BASE_URL}/models", timeout=3).status_code == 200
    except Exception:
        return False


def _is_rag_alive() -> bool:
    try:
        return requests.get(
            f"{SEARCH_API_URL}/search",
            params={"query": "test", "k": 1},
            timeout=3,
        ).status_code == 200
    except Exception:
        return False


def _get_current_model() -> str:
    try:
        resp  = requests.get(f"{VLLM_BASE_URL}/models", timeout=3).json()
        models = [m["id"] for m in resp.get("data", [])]
        base   = [m for m in models if m not in (MODEL_COT, MODEL_SFT)]
        return base[0] if base else "unknown"
    except Exception:
        return "unknown"


def _retrieve(question: str) -> list[str]:
    try:
        resp = requests.get(
            f"{SEARCH_API_URL}/search",
            params={"query": question, "k": TOP_K},
            timeout=15,
        )
        return resp.json().get("results", []) if resp.status_code == 200 else []
    except Exception as e:
        logger.error("RAG 검색 실패: %s", e)
        return []


def _build_user_content(question: str, context: str) -> str:
    return (
        f"### [도서관 정보]\n{context}\n\n"
        f"### [질문]\n{question}\n\n"
        f"### [지시 사항]\n"
        f"1. 친절한 말투로 규정에 근거하여 답변할 것.\n"
        f"2. 3문단 이내로 답변할 것.\n"
        f"3. 답변 끝에 지시 사항을 반복하지 말 것."
    )


# ─────────────────────────────────────────────────────────
# FastAPI 앱
# ─────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    vllm_ok = _is_vllm_alive()
    rag_ok  = _is_rag_alive()
    logger.info("🚀 서버 시작 | vLLM: %s | RAG: %s", "✅" if vllm_ok else "❌", "✅" if rag_ok else "❌")
    if not vllm_ok:
        logger.warning("⚠️ vLLM 서버 미응답. exaone 컨테이너를 확인하세요.")
    yield
    logger.info("🛑 서버 종료")


app = FastAPI(title="도서관 AI 사서 서버", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────
# 스키마
# ─────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    mode:     str              # baseline | cot | sft
    question: str
    context:  Optional[str] = ""
    use_rag:  bool           = False


# ─────────────────────────────────────────────────────────
# 엔드포인트
# ─────────────────────────────────────────────────────────

@app.get("/status")
def get_status():
    vllm_ok = _is_vllm_alive()
    return {
        "vllm":          vllm_ok,
        "rag":           _is_rag_alive(),
        "current_model": _get_current_model() if vllm_ok else "unavailable",
    }


@app.post("/query")
def query(req: QueryRequest):
    if not _is_vllm_alive():
        raise HTTPException(status_code=503, detail="vLLM 서버가 준비되지 않았습니다. exaone 컨테이너를 확인하세요.")

    client         = OpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)
    retrieved_docs = []

    # ── RAG 검색 (use_rag=True일 때 모든 모드에 적용) ─────
    if req.use_rag:
        if not _is_rag_alive():
            raise HTTPException(status_code=503, detail="RAG 검색 서버가 준비되지 않았습니다.")
        retrieved_docs = _retrieve(req.question)

    # ── 모드별 메시지 / model_id 구성 ─────────────────────
    if req.mode == "baseline":
        if req.use_rag:
            context_text = "\n\n".join(retrieved_docs) if retrieved_docs else "관련 참고 자료가 없습니다."
            messages = [
                {"role": "system", "content": SYSTEM_LIBRARY},
                {"role": "user",   "content": _build_user_content(req.question, context_text)},
            ]
        else:
            messages = [
                {"role": "system", "content": SYSTEM_BASELINE},
                {"role": "user",   "content": req.question},
            ]
        model_id = _get_current_model()

    elif req.mode in ("cot", "sft"):
        if req.use_rag:
            # RAG 검색 결과 + 수동 입력 context 병합
            rag_text     = "\n\n".join(retrieved_docs) if retrieved_docs else ""
            manual_text  = req.context.strip() if req.context and req.context.strip() else ""
            context_text = "\n\n".join(filter(None, [rag_text, manual_text])) or "도서관 운영 규정에 따라 답변합니다."
        else:
            context_text = req.context.strip() if req.context and req.context.strip() else "도서관 운영 규정에 따라 답변합니다."

        messages = [
            {"role": "system", "content": SYSTEM_LIBRARY},
            {"role": "user",   "content": _build_user_content(req.question, context_text)},
        ]
        model_id = MODEL_COT if req.mode == "cot" else MODEL_SFT

    else:
        raise HTTPException(status_code=400, detail=f"알 수 없는 mode: {req.mode}")

    # ── 추론 ─────────────────────────────────────────────
    start = time.perf_counter()
    try:
        res = client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=0,
            max_tokens=MAX_NEW_TOKENS,
        )
        elapsed = time.perf_counter() - start
        answer  = (res.choices[0].message.content or "").strip()
    except Exception as e:
        logger.error("추론 오류: %s", e)
        raise HTTPException(status_code=500, detail=f"추론 오류: {e}")

    logger.info("✅ [%s%s] %.2fs | %s...", req.mode, "+RAG" if req.use_rag else "", elapsed, req.question[:30])
    return {
        "answer":         answer,
        "elapsed":        round(elapsed, 3),
        "retrieved_docs": retrieved_docs,
        "model_id":       model_id,
    }