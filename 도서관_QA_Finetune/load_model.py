from huggingface_hub import snapshot_download
save_path = "/home/vsc/LLM/model/Exaone-3.5-32B-Instruct"

snapshot_download(
    repo_id="LGAI-EXAONE/EXAONE-3.5-32B-Instruct",
    local_dir=save_path,
    local_dir_use_symlinks=False,  # 실제 파일을 해당 경로에 직접 복사/다운로드
    resume_download=True, # 연결 중단 지점부터 다시 다운로드
    max_workers=8 # 병렬 다운로드 속도 향상
)
