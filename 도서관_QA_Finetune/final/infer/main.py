import sys
import os
import glob
import pandas as pd
import wandb
from config import Config
from datasets import Dataset
from model_utils import load_or_download_model_tokenizer
from data_augmentor import DataAugmentor

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from logger_config import setup_logger

logger = setup_logger()

def main() :
    logger.info("🏁 전체 프로세스 시작")
    config = Config()

    wandb.init(
        project="library-qa-finetune",
        name="data-augmentation"
    )

    # load dataset
    file_path = glob.glob("../data/*.xlsx")
    df = pd.read_excel(file_path[0])
    dataset = Dataset.from_pandas(df)

    logger.info(dataset)

    model, tokenizer = load_or_download_model_tokenizer(config)

    # data_augmentor.py의 run_augmentation 함수 호출
    augmentor = DataAugmentor(model, tokenizer, config)
    output_file = augmentor.run_augmentation(dataset)
    
    artifact = wandb.Artifact(f"{config.MODEL_ID}_{wandb.run.id}", type='dataset')
    artifact.add_file(output_file)
    wandb.log_artifact(artifact)
    wandb.finish()

    logger.info(f"🎉 데이터 증강 완료! 결과 파일: {output_file}")

if __name__ == "__main__" :
    main()