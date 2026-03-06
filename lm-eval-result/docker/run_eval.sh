#!/bin/bash
# run_eval.sh

# 결과 저장 폴더 생성
mkdir -p /app/result
LOG_FILE="/app/log/${MODEL_NAME}-$(date +%Y%m%d_%H%M%S).log"

echo "================================================"
echo "🚀 LLM Evaluation Framework Starting"
echo "📂 Model: ${MODEL_NAME}"
echo "📊 Tasks: ${TASKS}"
echo "🎯 Limit: ${LIMIT} samples per task"
echo "🧠 Few-shot: ${FEWSHOT}"
echo "⚡ GPU Util: ${GPU_UTIL}"
echo "================================================"

lm_eval --model vllm \
    --model_args "pretrained=${MODEL_BASE_DIR}/${MODEL_NAME},dtype=auto,trust_remote_code=True,gpu_memory_utilization=${GPU_UTIL}" \
    --tasks "${TASKS}" \
    --num_fewshot "${FEWSHOT}" \
    --batch_size 128 \
    --device cuda:0 \
    --limit "${LIMIT}" \
    --output_path "/app/result/${MODEL_NAME}-results.json" 2>&1 | tee -a "$LOG_FILE"

echo "================================================"
echo "✅ Evaluation Finished!"
echo "📄 Results saved to: /app/result/${MODEL_NAME}-results.json"
echo "================================================"