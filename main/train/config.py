import os
from datetime import datetime
import pytz
from transformers import BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
import torch

class Config :
    MODEL_ID = "LGAI-EXAONE/Exaone-3.5-7.8B-Instruct" # 학습 모델
    LOCAL_MODEL_DIR = "../../../../LLM/model/Exaone-3.5-7.8B-Instruct" # 학습 모델이 있는 디렉토리
    ADAPTER_PATH = "../SFT"

    kst = pytz.timezone('Asia/Seoul')
    now = datetime.now(kst).strftime("%Y%m%d_%H%M%S")

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
    DATASET_PATH = "../data/custom_data/augmented_cot_data_targeted_big.jsonl"
    TRAIN_GOLDEN_DATASET_PATH = "../data/golden_data/train_goldens.jsonl"
    VAL_GOLDEN_DATASET_PATH = "../data/golden_data/val_goldens.jsonl"
    TEST_GOLDEN_DATASET_PATH = "../data/golden_data/test_goldens.jsonl"
    # OUTPUT_DIR = "./SFT/Targeted_CoT_Big"
    OUTPUT_DIR = f"{ADAPTER_PATH}/{now}_SFT_with_Golden_Dataset"
    TRAINING_ARGS = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        fp16=False,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=0,
        max_length=1024, 
        packing=False,
        logging_steps=10,
        logging_strategy="steps",
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=100,
        report_to="wandb"
    )
