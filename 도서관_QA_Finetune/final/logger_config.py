import logging
import sys
from datetime import datetime, timedelta
import os
import time

def cleanup_old_logs(log_dir, days_to_keep=3):
    """
    지정된 디렉토리 내에서 days_to_keep보다 오래된 .log 파일을 삭제합니다.
    """
    if not os.path.exists(log_dir):
        return

    now = time.time()
    cutoff = now - (days_to_keep * 86400) # 86400초 = 1일

    try:
        for filename in os.listdir(log_dir):
            if filename.endswith(".log"):
                file_path = os.path.join(log_dir, filename)
                # 파일의 마지막 수정 시간(mtime) 확인
                if os.path.getmtime(file_path) < cutoff:
                    os.remove(file_path)
                    # print(f"🗑️ 오래된 로그 삭제됨: {filename}") # 디버깅용
    except Exception as e:
        print(f"⚠️ 로그 삭제 중 오류 발생: {e}")

def setup_logger(name=__name__, log_file=None, level=logging.INFO, days_to_keep=7):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.hasHandlers():
        return logger

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            # 🌟 로거 설정 시점에 오래된 로그 정리 실행
            cleanup_old_logs(log_dir, days_to_keep=days_to_keep)

        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger

# --- 아래 Getter 함수들에도 days_to_keep 파라미터를 추가하면 더 유연합니다 ---

def get_train_logger(base_dir='../log/train', days_to_keep=7):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return setup_logger(
        name='train_logger', 
        log_file=os.path.join(base_dir, f'train_{timestamp}.log'),
        days_to_keep=days_to_keep
    )

def get_infer_logger(base_dir='../log/infer', days_to_keep=7):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return setup_logger(
        name='infer_logger',
        log_file=os.path.join(base_dir, f'infer_{timestamp}.log'),
        days_to_keep=days_to_keep
    )

def get_eval_logger(base_dir='../log/eval', days_to_keep=7):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return setup_logger(
        name='eval_logger',
        log_file=os.path.join(base_dir, f'eval_{timestamp}.log'),
        days_to_keep=days_to_keep
    )