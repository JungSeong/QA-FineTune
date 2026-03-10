import sys
import os
import subprocess

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from base import generate_and_save_goldens_base
from evolution import generate_and_save_goldens_evolution
from split_golden_dataset import build_train_val_test_dataset
from logger_config import get_data_logger

logger = get_data_logger()

def main() :
    logger.info("⚒️ Base 데이터 생성 프로세스 시작")
    generate_and_save_goldens_base()
    logger.info("⚒️ Evolution 데이터 생성 프로세스 시작")
    generate_and_save_goldens_evolution()
    logger.info("✨ Golden train/val/test 데이터 생성 프로세스 시작")
    build_train_val_test_dataset()

if __name__ == "__main__" :
    main()