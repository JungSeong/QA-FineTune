import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from huggingface_hub import snapshot_download

save_path = "/home/vsc/LLM/model/Exaone-3.5-7.8B-Instruct-AWQ"

snapshot_download(
    repo_id="LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct-AWQ",
    local_dir=save_path,
    local_dir_use_symlinks=False,  # 실제 파일을 해당 경로에 직접 복사/다운로드
    max_workers=8 # 병렬 다운로드 속도 향상
)
