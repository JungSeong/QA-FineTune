import subprocess
import time
import sys
import os

embed_env = os.environ.copy()
embed_env["VLLM_USE_V1"] = "0"
embed_env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

def run_vllm_servers(model):
    # 1. Exaone LLM + LoRA 서버
    # --model 뒤의 경로와 --lora-modules의 경로를 컨테이너 내부 절대경로로 명시
    exaone_cmd = [
        "python3", "-m", "vllm.entrypoints.openai.api_server",
        "--model", f"/models/{model}",
        "--quantization", "bitsandbytes",
        "--dtype", "auto",
        "--max-model-len", "4096",
        "--gpu-memory-utilization", "0.8",
        "--trust-remote-code",
        "--port", "8000",
        # "--enable-lora",
        # "--max-loras", "2",
        # "--lora-modules", 
        # "excel=/models/Exaone-3.5-7.8B-Instruct-Basic/final_10",
        # "sft-lora=/models/Exaone-3.5-7.8B-Instruct-SFT-Golden/final_10"
        # "sft-lora=/models/20260312_132127_SFT_with_Golden/final_10"
        # "sft-lora=/models/20260312_172429_SFT_with_Golden/final_10"
    ]

    # 2. Embedding 서버
    embed_cmd = [
        "python3", "/app/embedding_server.py",
        "--model", "/embeddings/dragonkue/snowflake-arctic-embed-l-v2.0-ko",
        "--port", "8001",
        "--device", "cuda",
    ]

    print("🚀 Starting Exaone LLM Server (Port 8000)...")
    p1 = subprocess.Popen(exaone_cmd)

    # 메모리 경합 방지를 위한 충분한 대기 시간
    print("⏳ Waiting 20s for LLM to load before starting Embedding server...")
    time.sleep(20)

    print("🚀 Starting Embedding Server (Port 8001)...")
    p2 = subprocess.Popen(embed_cmd)

    try:
        p1.wait()
        p2.wait()
    except KeyboardInterrupt:
        p1.terminate()
        p2.terminate()

def run_judgement_servers():
    # 1. Exaone LLM + LoRA 서버
    # --model 뒤의 경로와 --lora-modules의 경로를 컨테이너 내부 절대경로로 명시
    judgement_cmd = [
        "python3", "-m", "vllm.entrypoints.openai.api_server",
        "--model", "/models/Qwen2.5-32B-Instruct-AWQ",
        "--dtype", "auto",
        "--max-model-len", "8192",
        "--gpu-memory-utilization", "0.8",
        "--trust-remote-code",
        "--port", "8000",
    ]

    # 2. Embedding 서버
    embed_cmd = [
        "python3", "-m", "vllm.entrypoints.openai.api_server",
        "--model", "/embeddings/dragonkue/snowflake-arctic-embed-l-v2.0-ko",
        "--served-model-name", "snowflake-arctic-embed-l-v2.0-ko",
        "--gpu-memory-utilization", "0.1",
        "--port", "8001"
    ]

    print("🚀 Starting Judgement LLM Server (Port 8000)...")
    p1 = subprocess.Popen(judgement_cmd)

    # 메모리 경합 방지를 위한 충분한 대기 시간
    print("⏳ Waiting 60s for LLM to load before starting Embedding server...")
    time.sleep(60)

    print("🚀 Starting Embedding Server (Port 8001)...")
    p2 = subprocess.Popen(embed_cmd, env=embed_env)

    try:
        p1.wait()
        p2.wait()
    except KeyboardInterrupt:
        p1.terminate()
        p2.terminate()

if __name__ == "__main__":
    func_name, situation = sys.argv[1], sys.argv[2]
    if func_name == "vllm" :
        if situation == "gen" :
            run_vllm_servers("Exaone-3.5-7.8B-Instruct")
        elif situation == "infer" :
            run_vllm_servers("Exaone-3.5-7.8B-Instruct")
    elif func_name == "judgement" :
        run_judgement_servers()