from transformers import BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
import torch

# 8000 -> A.X || 8001 -> EXAONE
class Config :
    GEN_MCP_URL = "http://localhost:8001/sse" 
    EVAL_MCP_URL = "http://localhost:8000/sse"
    MODEL_ID = "LGAI-EXAONE/Exaone-3.5-7.8B-Instruct"
    LOCAL_MODEL_DIR = "/home/vsc/LLM/model/Exaone-3.5-7.8B-Instruct"
    GEN_SERVER_MODEL_NAME = "Exaone-3.5-7.8B-Instruct"
    GEN_HF_MODEL_ID = "LGAI-EXAONE/Exaone-3.5-7.8B-Instruct"
    GEN_SAFE_MODEL_NAME = "Exaone-3.5-7.8B-Instruct"
    EVAL_MODEL_PATH = "/home/vsc/LLM/model/Qwen2.5-72B-Instruct-AWQ"
    EVAL_SERVER_MODEL_NAME = "Qwen2.5-72B-Instruct-AWQ"
    EVAL_HF_MODEL_ID = "Qwen/Qwen2.5-72B-Instruct-AWQ"
    EVAL_SAFE_MODEL_NAME = "Qwen2.5-72B-Instruct-AWQ"
    ADAPTER_PATH = "../train/SFT"
    INFER_DATA_PATH="/home/vsc/LLM_TUNE/QA-FineTune/main/data/evaluation/raw_generated_base.jsonl"
    EVALUATION_DATA_PATH="/home/vsc/LLM_TUNE/QA-FineTune/main/data/evaluation/final_evaluated_base.jsonl"
    MAX_LENGTH = 2048
    MAX_NEW_TOKENS = 2048
    TEMPERATURE=0.7
    QUANTIZATION_CONFIG = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4"
    )
    PEFT_CONFIG = LoraConfig(
        task_type="CAUSAL_LM",
        inference_mode=False,
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"]
    )
    DATASET_PATH = "../data/json/augmented_data.jsonl"
    RAW_DATASET_PATH="../../data/고양시도서관 FAQ1.xlsx"
    BENCHMARK_PATH = "../benchmark"
    LABELS = ["yes", "no", "info", "false"]