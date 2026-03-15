import os
from deepeval.synthesizer.config import EvolutionConfig
from deepeval.synthesizer import Synthesizer, Evolution
from deepeval.synthesizer.config import StylingConfig

class Config :
    XLSX_PATH   = "./raw/고양시도서관 FAQ1.xlsx"
    TXT_PATH    = "./raw/고양시도서관_FAQ1.txt"
    SAVE_DIR    = "./synthetic_data"
    BASE_FILE_NAME = "synthetic_goldens.jsonl"
    EVOLUTION_FILE_NAME   = "synthetic_goldens_evolution.jsonl"
    LLM_MODEL_NAME   = "/models/Exaone-3.5-7.8B-Instruct"
    LLM_BASE_URL     = "http://localhost:8002/v1"
    EMBED_MODEL_NAME = "snowflake-arctic-embed-l-v2.0-ko"
    EMBED_BASE_URL   = "http://localhost:8003/v1"
    MAX_GOLDENS_PER_CTX = 2
    MIN_CONTEXT_LENGTH  = 30
    MAX_RETRIES         = 3
    TIMEOUT_SECONDS     = 120.0
    CHUNK_SIZE          = 10       # 한 번에 처리할 컨텍스트 수
    STYLING_CONFIG = StylingConfig(
        input_format=(
            "고양시 도서관을 이용하는 시민이 사서에게 묻는 자연스러운 한국어 질문. "
            "구체적이고 실생활과 관련된 질문이어야 함."
        ),
        expected_output_format=(
            "4~5문장으로 간결하되 반드시핵심 내용을 담아 답변. "
            "말투는 '~에요!', '~입니다!'등 밝고 명량하게 답변. "
            "답변 앞에 어떠한 설명이나 제목도 붙이지 말 것. "
            "반드시 도서관 규정에 근거하여 정확하게 안내할 것."
        ),
        task="고양시 도서관 FAQ 기반 질의응답",
        scenario="도서관 이용 시민이 사서에게 이용 방법, 대출, 반납, 프로그램 등을 문의하는 상황",
    )
    EVOLUTION_CONFIG = EvolutionConfig(
        evolutions={
            Evolution.CONCRETIZING: 0.3,   # 구체적인 개념 질문
            Evolution.REASONING:    0.3,   # 논리 추론 질문
            Evolution.CONSTRAINED:  0.2,   # 특정 조건 / 제약 포함 질문
            Evolution.HYPOTHETICAL: 0.1,   # 가상 시나리오 질문
            Evolution.CONCRETIZING: 0.1    # 추상 개념 -> 구체적 사례 질문
        },
        num_evolutions=2,
    )
    INPUT_FILES = [
        f"{SAVE_DIR}/{BASE_FILE_NAME}",
        f"{SAVE_DIR}/{EVOLUTION_FILE_NAME}",
    ]
    RAW_OUTPUT_DIR= "./raw"
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
