import os
import logging
import torch
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download
from peft import prepare_model_for_kbit_training, get_peft_model
from config import Config

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from logger_config import setup_logger

logger = setup_logger()

def load_or_download_model_tokenizer(config):
    # 1. 디렉토리 확인 및 다운로드
    if not os.path.exists(config.LOCAL_MODEL_DIR) or not os.listdir(config.LOCAL_MODEL_DIR):
        logger.info(f"📡 모델이 {config.LOCAL_MODEL_DIR}에 없습니다. 다운로드를 시작합니다...")
        try : 
            snapshot_download(repo_id=config.MODEL_ID, local_dir=config.LOCAL_MODEL_DIR)
            logger.info("✅ 모델 다운로드 완료!")
        except Exception as e :
            logger.error(f"❌ 모델 다운로드 실패: {e}")
            raise e
    else:
        logger.info(f"📂 로컬 모델을 발견했습니다: {config.LOCAL_MODEL_DIR}")

    # 2. 모델 로드
    logger.info("🚀 모델 로딩 중...")
    tokenizer = AutoTokenizer.from_pretrained(config.LOCAL_MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        config.LOCAL_MODEL_DIR,
        quantization_config=config.QUANTIZATION_CONFIG,
        trust_remote_code=True,
        device_map="auto", # Accelerate가 자동 관리하도록 설정
        torch_dtype=torch.bfloat16
    )

    if hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
        model.get_input_embeddings = lambda: model.transformer.wte

    logger.info("✍🏿 Applying PEFT...")
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, config.PEFT_CONFIG)
    trainable_params, all_param = model.get_nb_trainable_parameters()
    logger.info(
        f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || "
        f"trainable%: {100 * trainable_params / all_param:.4f}"
    )
    
    # 패딩 토큰 설정 (생성 작업 필수)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer