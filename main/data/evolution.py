"""
고양시 도서관 FAQ 합성 데이터 생성기
- EvolutionConfig 기반 다양한 질문 유형 생성
- 청크 단위 처리 + 즉시 JSONL 저장
"""

import os
import re
import sys
import json
import logging
import pandas as pd
from pathlib import Path
from typing import List
from openai import OpenAI, AsyncOpenAI, APIConnectionError, APIStatusError, APITimeoutError
from deepeval.models import DeepEvalBaseLLM, DeepEvalBaseEmbeddingModel
from deepeval.synthesizer import Synthesizer, Evolution
from deepeval.synthesizer.config import EvolutionConfig
from config import Config

# ─────────────────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────────────────

config = Config()

XLSX_PATH   = config.XLSX_PATH
TXT_PATH    = config.TXT_PATH
SAVE_DIR    = config.SAVE_DIR
EVOLUTION_FILE_NAME = config.EVOLUTION_FILE_NAME
LLM_MODEL_NAME   = config.LLM_MODEL_NAME
LLM_BASE_URL     = config.LLM_BASE_URL
EMBED_MODEL_NAME = config.EMBED_MODEL_NAME
EMBED_BASE_URL   = config.EMBED_BASE_URL
MAX_GOLDENS_PER_CTX = config.MAX_GOLDENS_PER_CTX
MIN_CONTEXT_LENGTH  = config.MIN_CONTEXT_LENGTH
MAX_RETRIES         = config.MAX_RETRIES
TIMEOUT_SECONDS     = config.TIMEOUT_SECONDS
CHUNK_SIZE          = config.CHUNK_SIZE
EVOLUTION_CONFIG = config.EVOLUTION_CONFIG
styling_config = config.STYLING_CONFIG

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from logger_config import get_data_logger

log_base_path = os.path.join(parent_dir, "log", "data")
logger = get_data_logger(base_dir=log_base_path)


# ─────────────────────────────────────────────────────────
# 1. Excel → TXT 변환
# ─────────────────────────────────────────────────────────

def convert_xlsx_to_txt(xlsx_path: str, txt_path: str) -> str:
    """Excel FAQ를 'Q: ...\nA: ...\n\n' 형식 TXT로 변환합니다."""
    try:
        df = pd.read_excel(xlsx_path)
    except FileNotFoundError:
        logger.error("Excel 파일을 찾을 수 없습니다: %s", xlsx_path)
        raise
    except Exception as e:
        logger.error("Excel 읽기 실패: %s", e, exc_info=True)
        raise

    logger.info("컬럼: %s", df.columns.tolist())
    logger.info("행 수: %d", len(df))

    title_col = df.columns[1]
    desc_col  = df.columns[2]

    lines, skipped = [], 0
    for idx, row in df.iterrows():
        q = str(row[title_col]).strip()
        a = str(row[desc_col]).strip()
        if q and a and q != "nan" and a != "nan":
            lines.append(f"Q: {q}\nA: {a}\n")
        else:
            skipped += 1
            logger.debug("행 %d 스킵: q=%r, a=%r", idx, q, a)

    if skipped:
        logger.warning("총 %d개 행이 빈 값으로 스킵되었습니다.", skipped)

    if not lines:
        logger.error(
            "변환할 FAQ 데이터가 없습니다. 컬럼 인덱스를 확인하세요. (title=%s, desc=%s)",
            title_col, desc_col,
        )
        raise ValueError("FAQ 데이터가 비어 있습니다.")

    content = "\n".join(lines)

    try:
        Path(txt_path).parent.mkdir(parents=True, exist_ok=True)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(content)
    except OSError as e:
        logger.error("TXT 저장 실패 (%s): %s", txt_path, e, exc_info=True)
        raise

    logger.info("저장 완료: %s (%d 글자)", txt_path, len(content))
    logger.info("미리보기:\n%s", content[:400])
    return content


# ─────────────────────────────────────────────────────────
# 2. vLLM LLM 래퍼
# ─────────────────────────────────────────────────────────

class VLLMModel(DeepEvalBaseLLM):
    """vLLM OpenAI-호환 엔드포인트를 DeepEval LLM으로 래핑합니다."""

    def __init__(
        self,
        model_name: str,
        base_url: str,
        max_retries: int = MAX_RETRIES,
        timeout: float = TIMEOUT_SECONDS,
    ):
        self.model_name   = model_name
        self.client       = OpenAI(
            base_url=base_url, api_key="vllm-dummy",
            max_retries=max_retries, timeout=timeout,
        )
        self.async_client = AsyncOpenAI(
            base_url=base_url, api_key="vllm-dummy",
            max_retries=max_retries, timeout=timeout,
        )

    def load_model(self): return self.client
    def get_model_name(self) -> str: return self.model_name

    @staticmethod
    def _is_response_schema(schema) -> bool:
        if schema is None:
            return False
        try:
            return list(getattr(schema, "model_fields", {}).keys()) == ["response"]
        except Exception:
            return False

    def _get_system_prompt(self, schema) -> str:
        persona = (
            "당신은 고양시 도서관의 친절한 AI 사서입니다. "
            "4~5문장으로 간결하되 반드시핵심 내용을 담아 답변. "
            "말투는 '~에요!', '~입니다!'등 밝고 명량한 말투로 답변. "
            "답변 앞에 '친절하고 공손한' 같은 문구는 절대 붙이지 마세요."
        )
        base = "모든 답변은 반드시 한국어로 작성하세요. "
        if self._is_response_schema(schema):
            return persona + base
        if schema is not None:
            return base + "반드시 유효한 JSON 형식으로만 응답하세요. 설명이나 마크다운을 포함하지 마세요."
        return base + "도움이 되는 어시스턴트입니다."

    def _safe_json_parse(self, content: str, schema=None):
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

        try:
            match = re.search(r"(\{.*\}|\[.*\])", content, re.DOTALL)
            if not match:
                raise ValueError("JSON 구조를 찾을 수 없습니다.")

            json_str = match.group(1).strip()
            open_b, close_b = json_str.count("{"), json_str.count("}")
            if open_b > close_b:
                json_str = json_str.rstrip().rstrip(",")
                if "[" in json_str and "]" not in json_str:
                    json_str += "]"
                json_str += "}" * (open_b - close_b)

            data = json.loads(json_str)
            return schema(**data) if schema else data

        except Exception as e:
            logger.warning("JSON 파싱 실패: %s | 원본(50자): %s", e, content[:50])
            if schema:
                try:
                    return schema(data=[])
                except Exception:
                    logger.error("스키마 폴백 생성 실패: %s", schema, exc_info=True)
                    return None
            return {"data": []}

    def _build_messages(self, prompt: str, schema) -> list:
        return [
            {"role": "system", "content": self._get_system_prompt(schema)},
            {"role": "user",   "content": prompt},
        ]

    def _check_response(self, res, prompt: str, schema):
        choice = res.choices[0]
        if choice.finish_reason == "length":
            logger.warning(
                "응답이 max_tokens에서 잘렸습니다 — JSON 불완전 가능성 있음 (completion_tokens: %d)",
                res.usage.completion_tokens if res.usage else -1,
            )
        elif choice.finish_reason not in ("stop", None):
            logger.warning("비정상 finish_reason: %s", choice.finish_reason)

        content = choice.message.content
        if not content or not content.strip():
            logger.error(
                "LLM 빈 응답 (finish_reason: %s, prompt 앞부분: %.80s...)",
                choice.finish_reason, prompt,
            )
            return schema(response="") if self._is_response_schema(schema) else {"data": []}

        return self._safe_json_parse(content, schema)

    def generate(self, prompt: str, schema=None, **kwargs):
        try:
            res = self.client.chat.completions.create(
                model=self.model_name,
                messages=self._build_messages(prompt, schema),
                temperature=0, max_tokens=3000,
            )
            return self._check_response(res, prompt, schema)
        except APITimeoutError:
            logger.error("LLM 요청 타임아웃 (모델: %s)", self.model_name)
            raise
        except APIConnectionError as e:
            logger.error("LLM 서버 연결 실패: %s", e)
            raise
        except APIStatusError as e:
            logger.error("LLM API 오류 [HTTP %d]: %s", e.status_code, e.message)
            raise

    async def a_generate(self, prompt: str, schema=None, **kwargs):
        try:
            res = await self.async_client.chat.completions.create(
                model=self.model_name,
                messages=self._build_messages(prompt, schema),
                temperature=0, max_tokens=3000,
            )
            return self._check_response(res, prompt, schema)
        except APITimeoutError:
            logger.error("LLM 비동기 요청 타임아웃 (모델: %s)", self.model_name)
            raise
        except APIConnectionError as e:
            logger.error("LLM 비동기 서버 연결 실패: %s", e)
            raise
        except APIStatusError as e:
            logger.error("LLM 비동기 API 오류 [HTTP %d]: %s", e.status_code, e.message)
            raise


# ─────────────────────────────────────────────────────────
# 3. vLLM Embedding 래퍼
# ─────────────────────────────────────────────────────────

class VLLMEmbedding(DeepEvalBaseEmbeddingModel):
    """vLLM OpenAI-호환 임베딩 엔드포인트를 DeepEval Embedding으로 래핑합니다."""

    def __init__(
        self,
        model_name: str,
        base_url: str,
        max_retries: int = MAX_RETRIES,
        timeout: float = TIMEOUT_SECONDS,
    ):
        self.model_name   = model_name
        self.client       = OpenAI(
            base_url=base_url, api_key="vllm-dummy",
            max_retries=max_retries, timeout=timeout,
        )
        self.async_client = AsyncOpenAI(
            base_url=base_url, api_key="vllm-dummy",
            max_retries=max_retries, timeout=timeout,
        )

    def load_model(self): return self.client
    def get_model_name(self) -> str: return self.model_name

    def _call_embed(self, texts):
        try:
            return self.client.embeddings.create(input=texts, model=self.model_name)
        except APITimeoutError:
            logger.error("Embedding 타임아웃 (입력 수: %d)", len(texts) if isinstance(texts, list) else 1)
            raise
        except APIConnectionError as e:
            logger.error("Embedding 서버 연결 실패: %s", e)
            raise
        except APIStatusError as e:
            logger.error("Embedding API 오류 [HTTP %d]: %s", e.status_code, e.message)
            raise

    async def _async_call_embed(self, texts):
        try:
            return await self.async_client.embeddings.create(input=texts, model=self.model_name)
        except APITimeoutError:
            logger.error("Embedding 비동기 타임아웃 (모델: %s)", self.model_name)
            raise
        except APIConnectionError as e:
            logger.error("Embedding 비동기 연결 실패: %s", e)
            raise
        except APIStatusError as e:
            logger.error("Embedding 비동기 API 오류 [HTTP %d]: %s", e.status_code, e.message)
            raise

    def embed_text(self, text: str) -> List[float]:
        return self._call_embed(text).data[0].embedding

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return [d.embedding for d in self._call_embed(texts).data]

    async def a_embed_text(self, text: str) -> List[float]:
        return (await self._async_call_embed(text)).data[0].embedding

    async def a_embed_texts(self, texts: List[str]) -> List[List[float]]:
        return [d.embedding for d in (await self._async_call_embed(texts)).data]


# ─────────────────────────────────────────────────────────
# 4. 컨텍스트 준비
# ─────────────────────────────────────────────────────────

def load_contexts(txt_path: str, min_length: int = MIN_CONTEXT_LENGTH) -> List[List[str]]:
    """단일 블록 컨텍스트 — CONCRETIZING, CONSTRAINED, REASONING용"""
    try:
        with open(txt_path, encoding="utf-8") as f:
            full_text = f.read().strip()
    except FileNotFoundError:
        logger.error("TXT 파일을 찾을 수 없습니다: %s", txt_path)
        raise

    contexts = [[b.strip()] for b in full_text.split("\n\n") if len(b.strip()) >= min_length]

    if not contexts:
        raise ValueError("컨텍스트가 비어 있습니다.")

    logger.info("단일 컨텍스트 수: %d", len(contexts))
    return contexts


def load_multi_contexts(txt_path: str, min_length: int = MIN_CONTEXT_LENGTH,
                        multi_size: int = 2) -> List[List[str]]:
    """다중 블록 컨텍스트 — MULTICONTEXT용"""
    try:
        with open(txt_path, encoding="utf-8") as f:
            full_text = f.read().strip()
    except FileNotFoundError:
        logger.error("TXT 파일을 찾을 수 없습니다: %s", txt_path)
        raise

    blocks   = [b.strip() for b in full_text.split("\n\n") if len(b.strip()) >= min_length]
    contexts = [blocks[i:i + multi_size] for i in range(0, len(blocks), multi_size)]

    if not contexts:
        raise ValueError("컨텍스트가 비어 있습니다.")

    logger.info("다중 컨텍스트 수: %d (블록 %d개씩 묶음)", len(contexts), multi_size)
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
    try:
        single = embedder.embed_text(sample_texts[0])
        logger.info("단일 임베딩 길이: %d | 첫 5개 값: %s", len(single), single[:5])
        batch = embedder.embed_texts(sample_texts)
        logger.info("배치 결과 개수: %d | 각 길이: %s", len(batch), [len(e) for e in batch])
    except Exception as e:
        logger.error("임베딩 검증 실패 — 서버 상태 및 모델 경로를 확인하세요: %s", e, exc_info=True)
        raise


# ─────────────────────────────────────────────────────────
# 6. Golden 생성 및 즉시 JSONL 저장 (청크 단위)
# ─────────────────────────────────────────────────────────

def _append_jsonl(path: Path, goldens: list) -> None:
    """golden 리스트를 JSONL 파일에 즉시 추가합니다."""
    with open(path, "a", encoding="utf-8") as f:
        for g in goldens:
            metadata = getattr(g, "additional_metadata", None) or {}
            evolution_type = metadata.get("evolution_type", "original")

            record = {
                "input":           g.input,
                "actual_output":   getattr(g, "actual_output", None),
                "expected_output": g.expected_output,
                "context":         g.context,
                "source_file":     getattr(g, "source_file", None),
                "evolution_type":  evolution_type,
                "additional_metadata": metadata,
            }
            logger.info("생성된 질문 유형: %-20s | input: %s", evolution_type, g.input[:60])
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def generate_and_save_goldens(
    synthesizer: Synthesizer,
    contexts: List[List[str]],
    save_dir: str,
    file_name: str = EVOLUTION_FILE_NAME,
    max_goldens_per_context: int = MAX_GOLDENS_PER_CTX,
    chunk_size: int = CHUNK_SIZE,
) -> None:
    """
    컨텍스트를 chunk_size 단위로 나눠 처리하고,
    각 청크 완료 즉시 JSONL 파일에 저장합니다.
    중간에 오류가 발생해도 이전 청크 결과는 보존됩니다.
    """
    try:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error("저장 디렉토리 생성 실패 (%s): %s", save_dir, e, exc_info=True)
        raise

    jsonl_path = Path(save_dir) / f"{file_name}"

    if jsonl_path.exists():
        logger.warning("기존 JSONL 파일을 초기화합니다: %s", jsonl_path)
        jsonl_path.unlink()

    total         = len(contexts)
    chunks        = [contexts[i:i + chunk_size] for i in range(0, total, chunk_size)]
    total_saved   = 0
    failed_chunks = []

    logger.info(
        "총 컨텍스트 %d개를 %d개 청크로 처리합니다 (청크 크기: %d)",
        total, len(chunks), chunk_size,
    )

    for chunk_idx, chunk in enumerate(chunks, start=1):
        logger.info("[청크 %d/%d] 컨텍스트 %d개 처리 시작", chunk_idx, len(chunks), len(chunk))

        try:
            goldens = synthesizer.generate_goldens_from_contexts(
                contexts=chunk,
                include_expected_output=True,
                max_goldens_per_context=max_goldens_per_context,
            )
        except Exception as e:
            logger.error(
                "[청크 %d/%d] Golden 생성 실패 — 건너뜁니다: %s",
                chunk_idx, len(chunks), e, exc_info=True,
            )
            failed_chunks.append(chunk_idx)
            continue

        if not goldens:
            logger.warning("[청크 %d/%d] 생성된 golden이 없습니다.", chunk_idx, len(chunks))
            continue

        try:
            _append_jsonl(jsonl_path, goldens)
        except OSError as e:
            logger.error("[청크 %d/%d] JSONL 저장 실패: %s", chunk_idx, len(chunks), e, exc_info=True)
            failed_chunks.append(chunk_idx)
            continue

        total_saved += len(goldens)
        logger.info(
            "[청크 %d/%d] %d개 저장 완료 (누적: %d개) → %s",
            chunk_idx, len(chunks), len(goldens), total_saved, jsonl_path,
        )

        # 첫 번째 청크 미리보기
        if chunk_idx == 1:
            g = goldens[0]
            logger.info(
                "\n=== 첫 번째 Golden 미리보기 ===\nInput: %s\nExpected Output: %s\nContext: %s",
                g.input, g.expected_output, g.context,
            )

    logger.info("파이프라인 완료 — 총 %d개 저장, 실패 청크: %s", total_saved, failed_chunks or "없음")
    if failed_chunks:
        logger.warning("실패한 청크 번호: %s — 해당 컨텍스트를 재시도하세요.", failed_chunks)


# ─────────────────────────────────────────────────────────
# 7. 메인
# ─────────────────────────────────────────────────────────

def generate_and_save_goldens_evolution() :
    try:
        # Step 1: Excel → TXT
        convert_xlsx_to_txt(XLSX_PATH, TXT_PATH)

        # Step 2: 모델 인스턴스 생성
        local_llm = VLLMModel(
            model_name=LLM_MODEL_NAME,
            base_url=LLM_BASE_URL,
            max_retries=MAX_RETRIES,
            timeout=TIMEOUT_SECONDS,
        )
        local_embedder = VLLMEmbedding(
            model_name=EMBED_MODEL_NAME,
            base_url=EMBED_BASE_URL,
            max_retries=MAX_RETRIES,
            timeout=TIMEOUT_SECONDS,
        )

        # Step 3: 임베딩 동작 확인
        verify_embedder(local_embedder)

        # Step 4: 컨텍스트 로드
        single_contexts = load_contexts(TXT_PATH)
        multi_contexts  = load_multi_contexts(TXT_PATH, multi_size=2)

        # Step 5: Synthesizer 생성 (EvolutionConfig 적용)
        synthesizer_single = Synthesizer(
            model=local_llm,
            evolution_config=EVOLUTION_SINGLE_CONFIG,
            styling_config=styling_config
        )

        synthesizer_multiple = Synthesizer(
            model=local_llm,
            evolution_config=EVOLUTION_MULTIPLE_CONFIG,
            styling_config=styling_config
        )

        # Step 6: 청크 단위 생성 + 즉시 JSONL 저장
        generate_and_save_goldens(
            synthesizer=synthesizer_single,
            contexts=single_contexts,
            save_dir=SAVE_DIR,
            file_name=EVOLUTION_FILE_NAME,
        )

        generate_and_save_goldens(
            synthesizer=synthesizer_multiple,
            contexts=multi_contexts,
            save_dir=SAVE_DIR,
            file_name=EVOLUTION_FILE_NAME,
        )

    except Exception as e:
        logger.critical(
            "파이프라인 실행 중 치명적 오류 발생 — 프로세스를 종료합니다: %s",
            e, exc_info=True,
        )
        sys.exit(1)