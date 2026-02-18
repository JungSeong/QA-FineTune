import logging
import sys
from datetime import datetime
import os

# 1. log_file의 기본값을 None으로 변경해야 에러가 나지 않습니다.
def setup_logger(name=__name__, log_file=None, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 이미 핸들러가 있다면(중복 호출 방지), 기존 로거 반환
    if logger.hasHandlers():
        return logger

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 2. 콘솔 핸들러 추가
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 3. 파일 핸들러 추가 (log_file 경로가 있을 때만 실행)
    if log_file:
        # (중요) 디렉토리가 없으면 FileHandler 생성 시 에러가 나므로, 먼저 만듭니다.
        log_dir = os.path.dirname(log_file)
        if log_dir: # "train.log" 처럼 경로 없이 파일명만 있는 경우 제외
            os.makedirs(log_dir, exist_ok=True)

        # 디렉토리 생성 후 핸들러 연결
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # (선택) 상위 로거로 전파 방지 (로그 중복 출력 방지)
    logger.propagate = False

    return logger

def get_train_logger(base_dir='../log/train'):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return setup_logger(
        name='train_logger', 
        log_file=os.path.join(base_dir, f'train_{timestamp}.log')
    )

def get_infer_logger(base_dir='../log/infer'):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return setup_logger(
        name='infer_logger',
        log_file=os.path.join(base_dir, f'infer_{timestamp}.log')
    )