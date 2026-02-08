nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,temperature.gpu,power.draw --format=csv -l 1 > gpu_log_EXAONE_32B_quant_vLLM.csv
