from transformers import BitsAndBytesConfig
from peft import LoraConfig
import torch

class Config :
    MODEL_ID = "LGAI-EXAONE/Exaone-3.5-32B-Instruct"
    LOCAL_MODEL_DIR = "/home/vsc/LLM/model/Exaone-3.5-32B-Instruct"
    DATA_PATH = "../data/고양시도서관 FAQ1.xlsx"
    QUANTIZATION_CONFIG = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    PEFT_CONFIG = LoraConfig(
        task_type="CAUSAL_LM",
        inference_mode=False,
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"]
    )
    MIN_SAMPLE_COUNT = 5
    MAX_AUG_ITERATIONS = 5
    MAX_NEW_TOKENS = 512
    MAX_INPUT_LENGTH = 1024
    TEMPERATURE = 0.7 # 낮을수록 일관성 높음, 높을수록 다양성 높음
    TOP_P = 0.9 # 누적 확률이 P 이상인 토큰만 선택
    TOP_K = 50 # 확률이 가장 높은 상위 TOP_K개만 남김
    REPETITION_PENALTY = 1.1 # 1.0은 패널티 없음, 높을수록 반복 감소
    NUM_RETURN_SEQUENCES = 3 # 생성할 샘플 수