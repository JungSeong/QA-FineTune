"""
직접 AUGMENT한 dataset, DeepEval Synthesizer로 AUGMENT한 Golden Dataset을
HuggingFace의 Dataset으로 만들어주는 함수
"""

import pandas as pd
import os
import sys
import json
from config import Config
from datasets import Dataset, DatasetDict

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from logger_config import get_train_logger

logger = get_train_logger()

def preprocess_golden_dataset():
    config = Config()

    required = {
        "train": config.TRAIN_GOLDEN_DATASET_PATH,
        "val":   config.VAL_GOLDEN_DATASET_PATH,
        "test":  config.TEST_GOLDEN_DATASET_PATH,
    }

    # 경로 존재 여부 일괄 확인
    missing = [name for name, path in required.items() if not path]
    if missing:
        logger.error("다음 데이터셋 경로가 설정되지 않았습니다: %s", missing)
        raise ValueError(f"누락된 경로: {missing}")

    def load_jsonl(path: str, split_name: str) -> Dataset:
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning("[%s:%d] JSON 파싱 실패 — 스킵: %s", split_name, line_no, e)

        if not records:
            raise ValueError(f"{split_name} 파일에서 읽은 레코드가 없습니다: {path}")

        df = pd.DataFrame(records)[["question", "answer", "original_title"]]
        logger.info("👑 %s 로드 완료: %d개 → %s", split_name, len(df), path)
        return Dataset.from_pandas(df, preserve_index=False)

    final_dataset = DatasetDict({
        split: load_jsonl(path, split)
        for split, path in required.items()
    })

    return final_dataset

def preprocess_dataset() :
    config = Config()
    data = []
    seed = 42

    if config.DATASET_PATH :
        logger.info("Loading Dataset...")
        with open(config.DATASET_PATH, "r", encoding="utf-8") as f :
            for line in f :
                data.append(json.loads(line))
        df = pd.DataFrame(data)
        hf_dataset = Dataset.from_pandas(df[["question", "answer", "original_title"]])

        ds_split = hf_dataset.train_test_split(test_size=0.2, seed=seed)
        train_val_split = ds_split["train"].train_test_split(test_size = 0.3, seed=seed)

        final_dataset = DatasetDict({
            'train': train_val_split['train'],
            'val': train_val_split['test'],
            'test': ds_split['test']
        })

        return final_dataset