"""
[Step 2] RAGAS 평가 — qwen 컨테이너에서 실행

1_infer.py가 저장한 JSONL을 읽어 RAGAS로 평가합니다.
판사 LLM으로 Qwen을 사용합니다.

평가 지표 (ragas 0.4.3):
  _AnswerCorrectness  : ground_truth 대비 정답 일치도
  _Faithfulness       : contexts 범위 내에서 답변했는지 (환각 탐지)
  _ResponseRelevancy  : 질문과 답변의 관련성
"""

import os
import sys
import json
import logging

import pandas as pd

from pathlib import Path
from typing import List, Dict, Tuple

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import _AnswerCorrectness, _Faithfulness, _ResponseRelevancy
from ragas.llms import LangchainLLMWrapper
from ragas.run_config import RunConfig
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from logger_config import get_eval_logger

logger = get_eval_logger()

# ─────────────────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────────────────

INFER_SAVE_DIR = "../infer/infer_results"
EVAL_SAVE_DIR  = "./eval_results"
EXP_FILES = [
    "exp1_baseline",
    "exp2_rag",
    "exp3_cot",
    "exp4_sft",
]
JUDGE_BASE_URL = "http://localhost:8002/v1"
JUDGE_MODEL    = "/models/Qwen2.5-32B-Instruct-AWQ"
EMBED_MODEL_PATH = "/home/vsc/LLM/embedding/dragonkue/snowflake-arctic-embed-l-v2.0-ko"
BATCH_SIZE  = 10
MAX_WORKERS = 16
TIMEOUT     = 1200
SEED = 42
METRIC_COLS = ["answer_correctness", "faithfulness", "answer_relevancy"]

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from logger_config import get_eval_logger
logger = get_eval_logger()

# ─────────────────────────────────────────────────────────
# 1. 로컬 임베딩 래퍼 (RAGAS BaseRagasEmbeddings 직접 구현)
# ─────────────────────────────────────────────────────────

from ragas.embeddings import BaseRagasEmbeddings
from sentence_transformers import SentenceTransformer
from typing import List

class LocalEmbeddings(BaseRagasEmbeddings):
    def __init__(self, model_path: str, device: str = "cpu"):
        # 🌟 핵심: RAGAS는 self.model이 '문자열'이길 기대합니다.
        self.model_name = model_path 
        self._transformer = SentenceTransformer(model_path, device=device)
        
    @property
    def model(self):
        # RAGAS 내부 검증 로직이 이 값을 읽어갑니다. 반드시 문자열을 반환해야 합니다.
        return self.model_name

    def embed_query(self, text: str) -> List[float]:
        return self._transformer.encode(text, normalize_embeddings=True).tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._transformer.encode(texts, normalize_embeddings=True).tolist()

    # RAGAS 0.4.x 호환성을 위한 비동기 메서드 추가
    async def aembed_query(self, text: str) -> List[float]:
        return self.embed_query(text)

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embed_documents(texts)


# ─────────────────────────────────────────────────────────
# 2. 추론 결과 로드
# ─────────────────────────────────────────────────────────

def load_infer_result(exp_name: str) -> List[Dict]:
    path = Path(INFER_SAVE_DIR) / f"{exp_name}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"추론 결과 파일 없음: {path}\n1_infer.py를 먼저 실행하세요.")
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    logger.info("%s 로드 완료: %d개", exp_name, len(records))
    return records


# ─────────────────────────────────────────────────────────
# 3. 속도 지표 계산
# ─────────────────────────────────────────────────────────

def compute_speed_metrics(records: List[Dict]) -> Dict:
    elapsed_list = [r["elapsed_sec"] for r in records]
    responses    = [r["response"]    for r in records]
    n            = len(elapsed_list)
    mean_sec     = sum(elapsed_list) / n
    mean_tps     = sum(
        len(resp.split()) / sec
        for resp, sec in zip(responses, elapsed_list) if sec > 0
    ) / n
    return {
        "mean_sec_per_sample": round(mean_sec, 3),
        "mean_tps":            round(mean_tps, 2),
    }


# ─────────────────────────────────────────────────────────
# 4. Judge 빌드
# ─────────────────────────────────────────────────────────

def build_judge() -> Tuple:
    logger.info("🔗 Judge LLM 연결 중: %s", JUDGE_MODEL)
    judge_llm = LangchainLLMWrapper(
        ChatOpenAI(
            base_url=JUDGE_BASE_URL,
            api_key="vllm-dummy",
            model=JUDGE_MODEL,
            temperature=0.01,
            timeout=600,
        )
    )

    logger.info("🤗 로컬 임베딩 모델 로드 중: %s", EMBED_MODEL_PATH)
    judge_emb = LocalEmbeddings(EMBED_MODEL_PATH)

    return judge_llm, judge_emb


# ─────────────────────────────────────────────────────────
# 5. RAGAS 평가
# ─────────────────────────────────────────────────────────

def build_ragas_dataset(records: List[Dict]) -> Dataset:
    return Dataset.from_list([
        {
            "question":     r["question"],
            "answer":       r["response"],
            "contexts":     r["contexts"] if isinstance(r["contexts"], list) else [r["contexts"]],
            "ground_truth": r["ground_truth"],
        }
        for r in records
    ])


def run_ragas_batched(
    records: List[Dict],
    judge_llm,
    judge_emb,
    exp_name: str,
) -> pd.DataFrame:
    """배치 단위로 RAGAS 평가 후 DataFrame 반환"""
    metrics = [
        _AnswerCorrectness(),
        _Faithfulness(),
        _ResponseRelevancy(),
    ]
    all_dfs = []

    for i in range(0, len(records), BATCH_SIZE):
        batch   = records[i:i + BATCH_SIZE]
        end_idx = i + len(batch)
        logger.info("[%s] 배치 %d~%d 평가 중...", exp_name, i + 1, end_idx)

        dataset = build_ragas_dataset(batch)
        result  = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=judge_llm,
            embeddings=judge_emb,
            run_config=RunConfig(
                max_workers=MAX_WORKERS,
                timeout=TIMEOUT,
            ),
        )
        batch_df = result.to_pandas()
        logger.info("✅ 배치 완료: %s", result)
        all_dfs.append(batch_df)

    return pd.concat(all_dfs, ignore_index=True)


# ─────────────────────────────────────────────────────────
# 6. 결과 저장
# ─────────────────────────────────────────────────────────

def save_results(all_results: Dict[str, Dict]) -> None:
    Path(EVAL_SAVE_DIR).mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for exp_name, result in all_results.items():
        # 상세 CSV 저장
        detail_path = Path(EVAL_SAVE_DIR) / f"detail_{exp_name}.csv"
        result["detail_df"].to_csv(detail_path, index=False, encoding="utf-8-sig")
        logger.info("상세 결과 저장: %s", detail_path)

        # 요약 행 구성
        row = {"experiment": exp_name}
        row.update(result.get("ragas_avg", {}))
        row.update(result.get("speed", {}))
        summary_rows.append(row)

    summary_df   = pd.DataFrame(summary_rows)
    summary_path = Path(EVAL_SAVE_DIR) / "summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    logger.info("요약 CSV 저장 완료: %s", summary_path)
    print("\n" + summary_df.to_string(index=False))


# ─────────────────────────────────────────────────────────
# 7. 메인
# ─────────────────────────────────────────────────────────

def main() -> None:
    try:
        judge_llm, judge_emb = build_judge()
        all_results = {}

        for exp_name in EXP_FILES:
            logger.info("=" * 55)
            logger.info("▶ %s 평가 시작", exp_name)

            records  = load_infer_result(exp_name)
            speed    = compute_speed_metrics(records)
            detail_df = run_ragas_batched(records, judge_llm, judge_emb, exp_name)

            # 컬럼명 확인 후 평균 계산
            available_cols = [c for c in METRIC_COLS if c in detail_df.columns]
            if len(available_cols) != len(METRIC_COLS):
                logger.warning(
                    "예상 컬럼 일부 없음. 실제 컬럼: %s", detail_df.columns.tolist()
                )
            ragas_avg = detail_df[available_cols].mean().round(4).to_dict()

            all_results[exp_name] = {
                "detail_df": detail_df,
                "ragas_avg": ragas_avg,
                "speed":     speed,
            }
            logger.info("%s 완료: %s", exp_name, {**ragas_avg, **speed})

        save_results(all_results)
        logger.info("✅ 모든 평가 완료 → %s", EVAL_SAVE_DIR)

    except Exception as e:
        logger.critical("평가 파이프라인 오류: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()