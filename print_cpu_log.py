import psutil
import time
import csv
from datetime import datetime

file_name = "cpu_detailed_log_EXAONE_32B_quant_vLLM.csv"

# CSV 헤더 작성
with open(file_name, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp", "cpu_usage_percent", "cpu_temp", "load_1min", "mem_usage_percent"])

print(f"Logging started... Saving to {file_name}")

try:
    while True:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cpu_usage = psutil.cpu_percent(interval=1)
        # 온도 정보 (지원되는 하드웨어인 경우)
        temps = psutil.sensors_temperatures()
        cpu_temp = temps['coretemp'][0].current if 'coretemp' in temps else "N/A"
        load_1, _, _ = psutil.getloadavg()
        mem_usage = psutil.virtual_memory().percent

        with open(file_name, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([now, cpu_usage, cpu_temp, load_1, mem_usage])
        
        # 1초 대기 (psutil.cpu_percent(interval=1)에서 이미 1초 대기함)
except KeyboardInterrupt:
    print("\nLogging stopped.")
