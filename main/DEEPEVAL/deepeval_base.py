"""
고양시 도서관 FAQ 합성 데이터 생성기
- Excel FAQ → TXT 변환
- vLLM 기반 LLM/Embedding 래퍼
- DeepEval Synthesizer를 이용한 Golden 데이터셋 생성 및 저장
"""

import os
import re
import json
import logging
import sys
import pandas as pd

from pathlib import Path
from typing import List
from openai import OpenAI, AsyncOpenAI
from deepeval.models import DeepEvalBaseLLM, DeepEvalBaseEmbeddingModel
from deepeval.synthesizer import Synthesizer

# ─────────────────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────────────────

XLSX_PATH   = "/home/vsc/LLM_TUNE/QA-FineTune/main/data/고양시도서관 FAQ1.xlsx"
TXT_PATH    = "/home/vsc/LLM_TUNE/QA-FineTune/main/data/고양시도서관_FAQ1.txt"
SAVE_DIR    = "./synthetic_data"

LLM_MODEL_NAME       = "/models/Exaone-3.5-32B-Instruct"
LLM_BASE_URL         = "http://localhost:8002/v1"

EMBED_MODEL_NAME     = "/embeddings/dragonkue/snowflake-arctic-embed-l-v2.0-ko"
EMBED_BASE_URL       = "http://localhost:8003/v1"

MAX_GOLDENS_PER_CTX  = 3
MIN_CONTEXT_LENGTH   = 30

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from logger_config import get_data_logger
logger = get_data_logger()


# ─────────────────────────────────────────────────────────
# 1. Excel → TXT 변환
# ─────────────────────────────────────────────────────────

def convert_xlsx_to_txt(xlsx_path: str, txt_path: str) -> str:
    """
    Excel FAQ 파일을 'Q: ...\nA: ...\n\n' 형식의 TXT로 변환하고,
    변환된 텍스트를 반환합니다.
    """
    df = pd.read_excel(xlsx_path)
    logger.info("컬럼: %s", df.columns.tolist())
    logger.info("행 수: %d", len(df))

    title_col = df.columns[1]  # 질문 컬럼
    desc_col  = df.columns[2]  # 답변 컬럼

    lines = []
    for _, row in df.iterrows():
        q = str(row[title_col]).strip()
        a = str(row[desc_col]).strip()
        if q and a and q != "nan" and a != "nan":
            lines.append(f"Q: {q}\nA: {a}\n")

    content = "\n".join(lines)

    Path(txt_path).parent.mkdir(parents=True, exist_ok=True)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(content)

    logger.info("저장 완료: %s (%d 글자)", txt_path, len(content))
    logger.info("미리보기:\n%s", content[:400])
    return content


# ─────────────────────────────────────────────────────────
# 2. vLLM LLM 래퍼
# ─────────────────────────────────────────────────────────

class VLLMModel(DeepEvalBaseLLM):
    """vLLM OpenAI-호환 엔드포인트를 DeepEval LLM으로 래핑합니다."""

    def __init__(self, model_name: str, base_url: str):
        self.model_name  = model_name
        self.client      = OpenAI(base_url=base_url, api_key="vllm-dummy")
        self.async_client = AsyncOpenAI(base_url=base_url, api_key="vllm-dummy")

    def load_model(self):
        return self.client

    def get_model_name(self) -> str:
        return self.model_name

    # ── 내부 헬퍼 ──────────────────────────────────────────

    @staticmethod
    def _is_response_schema(schema) -> bool:
        if schema is None:
            return False
        try:
            fields = getattr(schema, "model_fields", {})
            return list(fields.keys()) == ["response"]
        except Exception:
            return False

    def _get_system_prompt(self, schema) -> str:
        base = "모든 답변은 반드시 한국어로 작성하세요. "
        if self._is_response_schema(schema):
            return base + "간결하고 직접적으로 답변하세요."
        if schema is not None:
            return base + "반드시 유효한 JSON 형식으로만 응답하세요. 설명이나 마크다운을 포함하지 마세요."
        return base + "도움이 되는 어시스턴트입니다."

    def _safe_json_parse(self, content: str, schema=None):
        """응답 문자열을 안전하게 파싱합니다."""
        # response 단순 스키마 처리
        if self._is_response_schema(schema):
            try:
                match = re.search(r"\{.*\}", content, re.DOTALL)
                if match:
                    data = json.loads(match.group(0))
                    if "response" in data:
                        return schema(response=str(data["response"]))
            except Exception:
                pass
            return schema(response=content.strip().strip('"').strip("'"))

        # 일반 JSON 처리
        try:
            match = re.search(r"(\{.*\}|\[.*\])", content, re.DOTALL)
            if not match:
                raise ValueError("JSON 구조를 찾을 수 없습니다.")

            json_str = match.group(1).strip()
            open_b   = json_str.count("{")
            close_b  = json_str.count("}")
            if open_b > close_b:
                json_str = json_str.rstrip().rstrip(",")
                if "[" in json_str and "]" not in json_str:
                    json_str += "]"
                json_str += "}" * (open_b - close_b)

            data = json.loads(json_str)
            return schema(**data) if schema else data

        except Exception as e:
            logger.warning("파싱 실패: %s | 응답 요약: %s...", e, content[:50])
            if schema:
                try:
                    return schema(data=[])
                except Exception:
                    return None
            return {"data": []}

    def _build_messages(self, prompt: str, schema) -> list:
        return [
            {"role": "system", "content": self._get_system_prompt(schema)},
            {"role": "user",   "content": prompt},
        ]

    # ── 동기 / 비동기 생성 ────────────────────────────────

    def generate(self, prompt: str, schema=None, **kwargs):
        res = self.client.chat.completions.create(
            model=self.model_name,
            messages=self._build_messages(prompt, schema),
            temperature=0,
            max_tokens=3000,
        )
        return self._safe_json_parse(res.choices[0].message.content, schema)

    async def a_generate(self, prompt: str, schema=None, **kwargs):
        res = await self.async_client.chat.completions.create(
            model=self.model_name,
            messages=self._build_messages(prompt, schema),
            temperature=0,
            max_tokens=3000,
        )
        return self._safe_json_parse(res.choices[0].message.content, schema)


# ─────────────────────────────────────────────────────────
# 3. vLLM Embedding 래퍼
# ─────────────────────────────────────────────────────────

class VLLMEmbedding(DeepEvalBaseEmbeddingModel):
    """vLLM OpenAI-호환 임베딩 엔드포인트를 DeepEval Embedding으로 래핑합니다."""

    def __init__(self, model_name: str, base_url: str):
        self.model_name   = model_name
        self.client       = OpenAI(base_url=base_url, api_key="vllm-dummy")
        self.async_client = AsyncOpenAI(base_url=base_url, api_key="vllm-dummy")

    def load_model(self):
        return self.client

    def get_model_name(self) -> str:
        return self.model_name

    def embed_text(self, text: str) -> List[float]:
        resp = self.client.embeddings.create(input=text, model=self.model_name)
        return resp.data[0].embedding

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        resp = self.client.embeddings.create(input=texts, model=self.model_name)
        return [d.embedding for d in resp.data]

    async def a_embed_text(self, text: str) -> List[float]:
        resp = await self.async_client.embeddings.create(
            input=text, model=self.model_name
        )
        return resp.data[0].embedding

    async def a_embed_texts(self, texts: List[str]) -> List[List[float]]:
        resp = await self.async_client.embeddings.create(
            input=texts, model=self.model_name
        )
        return [d.embedding for d in resp.data]


# ─────────────────────────────────────────────────────────
# 4. 컨텍스트 준비
# ─────────────────────────────────────────────────────────

def load_contexts(txt_path: str, min_length: int = MIN_CONTEXT_LENGTH) -> List[List[str]]:
    """TXT 파일에서 FAQ 블록을 읽어 컨텍스트 리스트로 반환합니다."""
    with open(txt_path, encoding="utf-8") as f:
        full_text = f.read().strip()

    blocks = full_text.split("\n\n")
    contexts = [[b.strip()] for b in blocks if len(b.strip()) >= min_length]
    logger.info("유효 컨텍스트 수: %d", len(contexts))
    return contexts

# ─────────────────────────────────────────────────────────
# 5. 임베딩 동작 확인
# ─────────────────────────────────────────────────────────

def verify_embedder(embedder: VLLMEmbedding) -> None:
    sample_texts = [
        "회원증을 대리발급 할 수 있나요?",
        "가족회원이 무엇인가요?",
        "도서 대출 기간은 얼마나 되나요?",
    ]

    single = embedder.embed_text(sample_texts[0])
    logger.info("단일 임베딩 길이: %d | 첫 5개 값: %s", len(single), single[:5])

    batch = embedder.embed_texts(sample_texts)
    logger.info("배치 결과 개수: %d | 각 길이: %s", len(batch), [len(e) for e in batch])


# ─────────────────────────────────────────────────────────
# 6. Golden 생성 및 저장
# ─────────────────────────────────────────────────────────

def generate_and_save_goldens(
    synthesizer: Synthesizer,
    contexts: List[List[str]],
    save_dir: str,
    max_goldens_per_context: int = MAX_GOLDENS_PER_CTX,
) -> None:
    goldens = synthesizer.generate_goldens_from_contexts(
        contexts=contexts,
        include_expected_output=True,
        max_goldens_per_context=max_goldens_per_context,
    )
    logger.info("생성된 golden 개수: %d", len(goldens))

    # 첫 번째 golden 미리보기
    if synthesizer.synthetic_goldens:
        g = synthesizer.synthetic_goldens[0]
        logger.info(
            "\n=== 첫 번째 Golden ===\nInput: %s\nExpected Output: %s\nContext: %s",
            g.input, g.expected_output, g.context,
        )

    # 저장
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    for file_type in ("json", "csv"):
        synthesizer.save_as(file_type=file_type, directory=save_dir)
        logger.info("%s 저장 완료: %s/synthetic_goldens.%s", file_type.upper(), save_dir, file_type)


# ─────────────────────────────────────────────────────────
# 7. 메인
# ─────────────────────────────────────────────────────────

def main() -> None:
    # Step 1: Excel → TXT
    convert_xlsx_to_txt(XLSX_PATH, TXT_PATH)

    # Step 2: 모델 인스턴스 생성
    local_llm = VLLMModel(model_name=LLM_MODEL_NAME, base_url=LLM_BASE_URL)
    local_embedder = VLLMEmbedding(model_name=EMBED_MODEL_NAME, base_url=EMBED_BASE_URL)

    # Step 3: 임베딩 동작 확인
    verify_embedder(local_embedder)

    # Step 4: 컨텍스트 로드
    contexts = load_contexts(TXT_PATH)

    # Step 5: Synthesizer 생성 및 Golden 생성 + 저장
    synthesizer = Synthesizer(model=local_llm)
    generate_and_save_goldens(synthesizer, contexts, SAVE_DIR)


if __name__ == "__main__":
    main()