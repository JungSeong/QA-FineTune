import pandas as pd
import json
from pathlib import Path
from config import Config

config = Config()

XLSX_PATH = config.XLSX_PATH
RAW_OUTPUT_DIR = config.RAW_OUTPUT_DIR
SYSTEM_LIBRARY = (
    "당신은 도서관 운영에 대한 전문적인 지식을 가진 인공지능 사서입니다. "
    "제공된 [도서관 정보]를 바탕으로 사용자의 질문에 정확하고 친절하게 답하십시오."
)

df = pd.read_excel(XLSX_PATH)

def build_user_content(question: str, context: str) -> str:
    return (
        f"### [도서관 정보]\n{context}\n\n"
        f"### [질문]\n{question}\n\n"
        f"### [지시 사항]\n"
        f"1. 친절한 말투로 규정에 근거하여 답변할 것.\n"
        f"2. 3문단 이내로 답변할 것.\n"
        f"3. 답변 끝에 지시 사항을 반복하지 말 것."
    )

records = []
for _, row in df.iterrows():
    question = str(row["TITLE"]).strip()
    answer   = str(row["DES"]).strip()
    records.append({
        "messages": [
            {"role": "system",    "content": SYSTEM_LIBRARY},
            {"role": "user",      "content": build_user_content(question, answer)},
            {"role": "assistant", "content": answer},
        ]
    })

out_path = Path(f"{RAW_OUTPUT_DIR}/sft_raw.jsonl")
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"✅ {len(records)}개 저장 완료: {out_path}")