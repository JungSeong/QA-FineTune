import sys
import os
import glob
import pandas as pd
import wandb
import torch
import sys
from trl import SFTTrainer, SFTConfig
from config import Config
from datasets import Dataset
from model_utils import load_or_download_model_tokenizer
from preprocess_dataset import preprocess_dataset, preprocess_golden_dataset, preprocess_raw_dataset
from accelerate import Accelerator
from prompts import generate_prompts
from functools import partial

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from logger_config import get_train_logger

logger = get_train_logger()

def main() :
    logger.info("💪 훈련 프로세스 시작")
    config = Config()

    wandb.init(
        project="library-qa-finetune-train",
        name="train-with-accelerator",
        config=config.__dict__
    )

    logger.info("Loading Dataset...")
    #dataset = preprocess_dataset()
    # dataset = preprocess_raw_dataset()
    dataset = preprocess_golden_dataset()
    train_dataset, val_dataset, test_dataset = dataset['train'], dataset['val'], dataset['test']
    logger.info("Dataset loaded successfully")
    logger.info(f"train_dataset size: {len(train_dataset)}")
    logger.info(f"val_dataset size: {len(val_dataset)}")
    logger.info(f"test_dataset size: {len(test_dataset)}")

    logger.info("🪙 Loading Model with LoRA and Tokenizer...")
    model, tokenizer = load_or_download_model_tokenizer()
    logger.info(model)

    training_args = config.TRAINING_ARGS
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
        processing_class=tokenizer,
        formatting_func=generate_prompts
    )
    logger.info(f"🏋️ Training_arguments : {training_args}")
 
    torch.load = partial(torch.load, weights_only=False)
    last_checkpoint = None
    if os.path.isdir(config.OUTPUT_DIR):
        # OUTPUT_DIR 내의 'checkpoint-'로 시작하는 폴더들을 검색
        checkpoints = glob.glob(os.path.join(config.OUTPUT_DIR, "checkpoint-*"))
        if checkpoints:
            # 가장 숫자가 높은(최신) 체크포인트를 선택
            last_checkpoint = max(checkpoints, key=os.path.getctime)
            logger.info(f"✅ 마지막 체크포인트 발견: {last_checkpoint}. 학습을 재개합니다.")
    
    if last_checkpoint:
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        logger.info("🚀 Starting training...")
        trainer.train()

    accelerator = trainer.accelerator

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        trainer.save_model(f"{config.OUTPUT_DIR}/final_10")
        tokenizer.save_pretrained(f"{config.OUTPUT_DIR}/final_10")
        logger.info(f"🎓 Trained finished! Model saved to {config.OUTPUT_DIR}")
        wandb.finish()

if __name__ == "__main__" :
    main()