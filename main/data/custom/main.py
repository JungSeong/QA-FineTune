import sys
import os
import glob
import pandas as pd
import wandb
import asyncio
from config import Config
from datasets import Dataset
from data_augmentor import AsyncDataAugmentor
from data_augmentor_cot import CoTAsyncDataAugmentor

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

def main() :
    logger.info("데이터 생성 프로제스 시작")
    config = Config()

    wandb.init(
        project="library-qa-finetune-evaluation",
        name="data-augmentation-pipeline",
        config=config.__dict__
    )

    # load dataset
    file_path = glob.glob("../data/*.xlsx")
    df = pd.read_excel(file_path[0])
    dataset = Dataset.from_pandas(df)

    load_or_download_model_tokenizer(config)

    # augmentor = AsyncDataAugmentor(config.GEN_HF_MODEL_ID, config)
    # final_file = asyncio.run(augmentor.run_pipeline_async(dataset, f"{config.AUGMENTED_DATA_PATH}"))
    
    CoTaugmentor = CoTAsyncDataAugmentor(config.GEN_HF_MODEL_ID, config)
    final_file = asyncio.run(CoTaugmentor.run_pipeline_async(dataset, f"{config.AUGMENTED_COT_DATA_PATH}"))
    
    if final_file and os.path.exists(final_file):
        if wandb.run:
            logger.info(f"📦 W&B Artifact 업로드 시작: {final_file}")
            
            # 1. Artifact 객체 생성 (이름, 타입 지정)
            artifact = wandb.Artifact(
                name="augmented-qa-dataset", 
                type="dataset",
                description="LLM을 통해 증강된 QA 데이터셋"
            )
            
            # 2. 파일 추가
            artifact.add_file(final_file)
            
            # 3. WandB 서버로 업로드(로그)
            wandb.run.log_artifact(artifact)
            logger.info("🚀 Artifact 업로드 완료!")
    else:
        logger.error("❌ 업로드할 최종 결과 파일이 생성되지 않았습니다. 파이프라인 로그를 확인하세요.")

if __name__ == "__main__" :
    main()