from pathlib import Path
p = Path(__file__).parent.parent / "docker" / ".env"
print(p)           # 경로 출력
print(p.exists())  # 실제 존재 여부