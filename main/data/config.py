import os
from deepeval.synthesizer.config import EvolutionConfig
from deepeval.synthesizer import Synthesizer, Evolution

class Config :
    XLSX_PATH   = "./raw/고양시도서관 FAQ1.xlsx"
    TXT_PATH    = "./raw/고양시도서관_FAQ1.txt"
    SAVE_DIR    = "./synthetic_data"
    BASE_FILE_NAME = "synthetic_goldens.jsonl"
    EVOLUTION_FILE_NAME   = "synthetic_goldens_evolution.jsonl"
    LLM_MODEL_NAME   = "/models/Exaone-3.5-32B-Instruct"
    LLM_BASE_URL     = "http://localhost:8002/v1"
    EMBED_MODEL_NAME = "/embeddings/dragonkue/snowflake-arctic-embed-l-v2.0-ko"
    EMBED_BASE_URL   = "http://localhost:8003/v1"
    MAX_GOLDENS_PER_CTX = 1       # evolution 특성상 1개씩이 품질에 유리
    MIN_CONTEXT_LENGTH  = 30
    MAX_RETRIES         = 3
    TIMEOUT_SECONDS     = 120.0
    CHUNK_SIZE          = 10       # 한 번에 처리할 컨텍스트 수
    EVOLUTION_CONFIG = EvolutionConfig(
        evolutions={
            Evolution.CONCRETIZING: 0.5,   # 구체적인 개념 질문
            Evolution.REASONING:    0.3,   # 논리 추론 질문
            Evolution.HYPOTHETICAL: 0.2,   # 가상 시나리오 질문
        },
        num_evolutions=2,
    )
    INPUT_FILES = [
        f"{SAVE_DIR}/{BASE_FILE_NAME}",
        f"{SAVE_DIR}/{EVOLUTION_FILE_NAME}",
    ]
    OUTPUT_DIR= "./golden_data"
    TRAIN_FILE_PATH="train_goldens.jsonl"
    VAL_FILE_PATH="val_goldens.jsonl"
    TEST_FILE_PATH="test_goldens.jsonl"
    TRAIN_RATIO = 0.6
    VAL_RATIO   = 0.2
    TEST_RATIO  = 0.2
    SEED = 42
    FIELDS = ("input", "expected_output", "context")
    FIELD_RENAME = {
        "input":           "question",
        "expected_output": "answer",
        "context":         "original_title",
    }
