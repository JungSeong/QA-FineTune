"""
두 JSONL 파일에서 레코드를 읽어 context 단위로 train / val / test 분리 후 저장합니다.

핵심 원칙:
  - 동일한 context에서 생성된 QA쌍은 반드시 같은 split에 속해야 함
  - train의 context가 val/test에 노출되지 않아야 공정한 평가가 가능함
"""

import json
import logging
import random
import sys

from collections import defaultdict
from pathlib import Path

# ─────────────────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────────────────

INPUT_FILES = [
    "./synthetic_data/synthetic_goldens_evolution.jsonl",
    "./synthetic_data/synthetic_goldens.jsonl",
]

OUTPUT_DIR  = "./golden_data"
TRAIN_FILE  = "train_goldens.jsonl"
VAL_FILE    = "val_goldens.jsonl"
TEST_FILE   = "test_goldens.jsonl"

TRAIN_RATIO = 0.6   # 60% 학습
VAL_RATIO   = 0.2   # 20% 검증 (early stopping / 하이퍼파라미터 튜닝)
TEST       = 0.2  # 20% 테스트

SEED = 42

FIELDS = ("input", "expected_output", "context")

FIELD_RENAME = {
    "input":           "question",
    "expected_output": "answer",
    "context":         "original_title",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
# 1. 레코드 로드 및 context 단위로 그룹핑
# ─────────────────────────────────────────────────────────

def load_and_group_by_context(input_files: list[str]) -> dict[str, list[dict]]:
    """
    레코드를 읽어 context를 키로 그룹핑합니다.
    context가 list[str]이므로 tuple로 변환해 dict 키로 사용합니다.
    반환: { context_key: [record, record, ...], ... }
    """
    context_groups: dict[str, list[dict]] = defaultdict(list)
    skipped = 0

    for file_path in input_files:
        path = Path(file_path)

        if not path.exists():
            logger.error("파일을 찾을 수 없습니다 — 건너뜁니다: %s", file_path)
            continue

        logger.info("읽기 시작: %s", file_path)
        file_read = 0

        with open(path, encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning("JSON 파싱 실패 [%s:%d]: %s", path.name, line_no, e)
                    skipped += 1
                    continue

                missing = [field for field in FIELDS if field not in record]
                if missing:
                    logger.warning(
                        "필드 누락으로 스킵 [%s:%d]: %s",
                        path.name, line_no, missing,
                    )
                    skipped += 1
                    continue

                context_key = tuple(record["context"])
                context_groups[context_key].append(
                    {FIELD_RENAME[f]: record[f] for f in FIELDS}
                )
                file_read += 1

        logger.info("%s → %d개 로드 완료", path.name, file_read)

    total_qa       = sum(len(v) for v in context_groups.values())
    total_contexts = len(context_groups)
    logger.info(
        "전체 로드 완료 — 고유 context: %d개 / QA쌍: %d개 (스킵: %d개)",
        total_contexts, total_qa, skipped,
    )
    return dict(context_groups)


# ─────────────────────────────────────────────────────────
# 2. context 단위로 train / val / test 분리 후 저장
# ─────────────────────────────────────────────────────────

def split_by_context_and_save(
    context_groups: dict[str, list[dict]],
    output_dir: str,
    train_file: str,
    val_file: str,
    test_file: str,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float   = VAL_RATIO,
    seed: int          = SEED,
) -> None:
    if not context_groups:
        logger.error("저장할 레코드가 없습니다.")
        raise ValueError("context_groups가 비어 있습니다.")

    if train_ratio + val_ratio >= 1.0:
        raise ValueError(
            f"train_ratio({train_ratio}) + val_ratio({val_ratio}) >= 1.0 — test set이 없어집니다."
        )

    random.seed(seed)
    context_keys = list(context_groups.keys())
    random.shuffle(context_keys)

    total        = len(context_keys)
    train_end    = int(total * train_ratio)
    val_end      = train_end + int(total * val_ratio)

    train_keys   = context_keys[:train_end]
    val_keys     = context_keys[train_end:val_end]
    test_keys    = context_keys[val_end:]

    train_set    = [qa for key in train_keys for qa in context_groups[key]]
    val_set      = [qa for key in val_keys   for qa in context_groups[key]]
    test_set     = [qa for key in test_keys  for qa in context_groups[key]]

    output_path  = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    def write_jsonl(data: list[dict], path: Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for record in data:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    try:
        write_jsonl(train_set, output_path / train_file)
        write_jsonl(val_set,   output_path / val_file)
        write_jsonl(test_set,  output_path / test_file)
    except OSError as e:
        logger.error("파일 저장 실패: %s", e, exc_info=True)
        raise

    logger.info(
        "분리 완료 (seed=%d, ratio=%.0f%%:%.0f%%:%.0f%%) — "
        "train: context %d개 / QA %d개 → %s | "
        "val:   context %d개 / QA %d개 → %s | "
        "test:  context %d개 / QA %d개 → %s",
        seed,
        train_ratio * 100, val_ratio * 100, (1 - train_ratio - val_ratio) * 100,
        len(train_keys), len(train_set), output_path / train_file,
        len(val_keys),   len(val_set),   output_path / val_file,
        len(test_keys),  len(test_set),  output_path / test_file,
    )


# ─────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────

def main() -> None:
    try:
        context_groups = load_and_group_by_context(INPUT_FILES)
        split_by_context_and_save(
            context_groups,
            OUTPUT_DIR,
            TRAIN_FILE,
            VAL_FILE,
            TEST_FILE,
        )
    except Exception as e:
        logger.critical("처리 중 치명적 오류: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()