"""
고양시 도서관 FAQ 합성 데이터 생성기 (Base + Evolution)

[사용 방법]
1. QA-FineTune/docker에서 docker compose up -d
2. python3 main.py
"""

import sys
import os
import subprocess
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from base import generate_and_save_goldens_base
from evolution import generate_and_save_goldens_evolution
from split_golden_dataset import build_train_val_test_dataset
from logger_config import get_data_logger

logger = get_data_logger()

def main() :
    total_start = time.perf_counter()
    t0 = time.perf_counter()
    logger.info("⚒️ Base 데이터 생성 프로세스 시작")
    generate_and_save_goldens_base()
    logger.info("✅ Evolution 완료 | %.1fs", time.perf_counter() - t0)

    logger.info("✨ Golden train/val/test 데이터 생성 프로세스 시작")
    t0 = time.perf_counter()
    logger.info("⚒️ Evolution 데이터 생성 프로세스 시작")
    generate_and_save_goldens_evolution()
    logger.info("✅ 데이터셋 분리 완료 | %.1fs", time.perf_counter() - t0)

    build_train_val_test_dataset()
    total = time.perf_counter() - total_start
    logger.info("🎉 전체 파이프라인 완료 | 총 소요 시간: %.1fs (%.1f분)", total, total / 60)

if __name__ == "__main__" :
    main()