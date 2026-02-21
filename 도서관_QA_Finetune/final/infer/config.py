from transformers import BitsAndBytesConfig
from peft import LoraConfig
import torch

class Config :
    MCP_URL = "http://localhost:8000/sse"
    LOCAL_MODEL_DIR = "/home/vsc/LLM/model"
    AUGMENTED_DATA_PATH = "../data/json/augmented_data.jsonl"
    DATA_PATH = "../data/고양시도서관 FAQ1.xlsx"
    GEN_SERVER_MODEL_NAME = "A.X-4.0-Light"
    GEN_HF_MODEL_ID = "skt/A.X-4.0-Light"
    GEN_SAFE_MODEL_NAME = "A.X-4.0-Light"
    EVAL_MODEL_PATH = "/home/vsc/LLM/model/Exaone-3.5-32B-Instruct"
    EVAL_SERVER_MODEL_NAME = "Exaone-3.5-32B-Instruct"
    EVAL_HF_MODEL_ID = "LGAI-EXAONE/EXAONE-3.5-32B-Instruct"
    EVAL_SAFE_MODEL_NAME = "Exaone-3.5-32B-Instruct"
    EVAL_LOG_PATH = "/home/vsc/LLM_TUNE/QA-FineTune/도서관_QA_Finetune/final/data/json/LLM_EVAL.json"
    MIN_SAMPLE_COUNT = 5
    MAX_AUG_ITERATIONS = 2
    MAX_NEW_TOKENS = 512
    MAX_INPUT_LENGTH = 1024
    TEMPERATURE = 0.7 # 낮을수록 일관성 높음, 높을수록 다양성 높음
    TOP_P = 0.9 # 누적 확률이 P 이상인 토큰만 선택
    TOP_K = 50 # 확률이 가장 높은 상위 TOP_K개만 남김
    REPETITION_PENALTY = 1.1 # 1.0은 패널티 없음, 높을수록 반복 감소
    NUM_RETURN_SEQUENCES = 3 # 생성할 샘플 수
    NUM_RETURN_TARGET_SEQUENCES = 1
    LABELS = ["yes", "no", "info", "false"]
    NUM_SEMAPHORES = 10