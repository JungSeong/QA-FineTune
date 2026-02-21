import sys
import os
import glob
import pandas as pd
import wandb
import asyncio
from config import Config
from datasets import Dataset
from model_utils import load_or_download_model_tokenizer
from data_augmentor import AsyncDataAugmentor

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from logger_config import get_infer_logger

logger = get_infer_logger()

def main() :
    logger.info("🏁 전체 프로세스 시작")
    config = Config()

    wandb.init(
        project="library-qa-finetune",
        name="data-augmentation-pipeline",
        config=config.__dict__
    )

    # load dataset
    file_path = glob.glob("../data/*.xlsx")
    df = pd.read_excel(file_path[0])
    dataset = Dataset.from_pandas(df)

    load_or_download_model_tokenizer(config)

    augmentor = AsyncDataAugmentor(config.MCP_URL, config.GEN_HF_MODEL_ID, config)
    final_file = asyncio.run(augmentor.run_pipeline_async(dataset, f"{config.AUGMENTED_DATA_PATH}"))
    
    artifact = wandb.Artifact(f"{config.GEN_SAFE_MODEL_NAME}_{wandb.run.id}", type='dataset')
    artifact.add_file(final_file)
    wandb.log_artifact(artifact)
    wandb.finish()

    logger.info(f"🎉 데이터 증강 및 보완 완료! 결과 파일: {final_file}")

if __name__ == "__main__" :
    main()