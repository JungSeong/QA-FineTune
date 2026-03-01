import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
hf_token = os.environ.get("hf_token")

from huggingface_hub import login
if hf_token :
    login(token=hf_token) 
else :
    print("No hf token")

from huggingface_hub import snapshot_download

save_path = "/home/vsc/LLM/model/Qwen2.5-72B-Instruct"

snapshot_download(
    repo_id="Qwen/Qwen2.5-72B-Instruct-AWQ",
    local_dir=save_path,
    local_dir_use_symlinks=False,  # 실제 파일을 해당 경로에 직접 복사/다운로드
    max_workers=8 # 병렬 다운로드 속도 향상
)
