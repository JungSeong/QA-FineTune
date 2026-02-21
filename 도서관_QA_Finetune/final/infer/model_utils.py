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

from logger_config import get_infer_logger

logger = get_infer_logger()

def load_or_download_model_tokenizer(config):
    # 1. 디렉토리 확인 및 다운로드
    address = f"{config.LOCAL_MODEL_DIR}/{config.GEN_SERVER_MODEL_NAME}"
    logger.info(address)
    if not os.path.exists(address) or not os.listdir(address):
        logger.info(f"📡 모델 {config.GEN_SERVER_MODEL_NAME} 이 {config.LOCAL_MODEL_DIR}에 없습니다. 다운로드를 시작합니다...")
        try : 
            snapshot_download(repo_id=config.GEN_HF_MODEL_ID, local_dir=config.LOCAL_MODEL_DIR)
            logger.info("✅ 모델 다운로드 완료!")
        except Exception as e :
            logger.error(f"❌ 모델 다운로드 실패: {e}")
            raise e
    else:
        logger.info(f"📂 {address}에서 로컬 모델을 발견했습니다")