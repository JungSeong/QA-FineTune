FROM scitrera/dgx-spark-vllm:0.14.0rc2-t5
WORKDIR /app
COPY requirements_ax.txt .
RUN pip install --no-cache-dir -r requirements_ax.txt
COPY mcp_vllm.py .
CMD ["python3", "mcp_vllm.py"]
