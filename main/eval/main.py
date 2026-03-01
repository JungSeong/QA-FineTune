import sys
import os
import glob
import pandas as pd
import wandb
import asyncio
from config import Config
from datasets import Dataset
from final_evaluator import QAPipeline

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from logger_config import get_eval_logger

logger = get_eval_logger()

def main() :
    logger.info("📋 전체 Evaluation 프로세스 시작")
    config = Config()

    wandb.init(
        project="library-qa-finetune",
        name="final-evaluation-pipeline",
        config=config.__dict__
    )

    # load dataset
    file_path = glob.glob("../data/*.xlsx")
    df = pd.read_excel(file_path[0])
    dataset = Dataset.from_pandas(df)

    CoTaugmentor = QAPipeline(config.GEN_HF_MODEL_ID, config)
    final_file = asyncio.run(CoTaugmentor.run_pipeline_async(dataset, f"{config.EVALUATION_DATA_PATH}"))
    
    if final_file and os.path.exists(final_file):
        if wandb.run:
            logger.info(f"📦 W&B Artifact 업로드 시작: {final_file}")
            
            # 1. Artifact 객체 생성 (이름, 타입 지정)
            artifact = wandb.Artifact(
                name="augmented-qa-evaluation", 
                type="dataset",
                description=f"{config.EVAL_SERVER_MODEL_NAME}모델로 평가 완료"
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