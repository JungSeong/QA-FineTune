from transformers import BitsAndBytesConfig
from peft import LoraConfig
import torch

class Config :
    MODEL_ID = "LGAI-EXAONE/Exaone-3.5-2.4B-Instruct"
    LOCAL_MODEL_DIR = "/home/vsc/LLM/model/Exaone-3.5-2.4B-Instruct"
    AUGMENTED_DATA_PATH = "./data/json/augmented_data.jsonl"
    DATA_PATH = "../data/고양시도서관 FAQ1.xlsx"
    QUANTIZATION_CONFIG = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
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