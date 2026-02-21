from transformers import BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
import torch

class Config :
    MODEL_ID = "LGAI-EXAONE/Exaone-3.5-2.4B-Instruct"
    LOCAL_MODEL_DIR = "/home/vsc/LLM/model/Exaone-3.5-2.4B-Instruct"
    ADAPTER_PATH = "../train/SFT"
    MAX_LENGTH = 2048
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
    BENCHMARK_PATH = "../benchmark"