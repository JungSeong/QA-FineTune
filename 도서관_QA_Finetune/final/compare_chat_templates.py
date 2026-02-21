"""
모델별 chat_template 비교 스크립트
사용법: python compare_chat_templates.py
"""

import os
from transformers import AutoTokenizer

# ==============================
# 비교할 모델 목록 (경로 또는 HuggingFace ID)
# ==============================
MODEL_BASE_DIR = os.getenv("MODEL_BASE_DIR", "/models")

MODELS = [
    "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct",
    "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",
    "LGAI-EXAONE/EXAONE-3.5-32B-Instruct",
    # 로컬 경로도 가능:
    # "exaone-2.4b",
    # "exaone-7.8b",
]

# 테스트용 메시지
TEST_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "안녕하세요, 도서관 운영 시간이 어떻게 되나요?"},
]

SEPARATOR = "=" * 80


def resolve_model_path(model_id: str) -> str:
    local_path = os.path.join(MODEL_BASE_DIR, model_id)
    if os.path.exists(local_path):
        return local_path
    return model_id


def analyze_model(model_id: str):
    path = resolve_model_path(model_id)
    print(f"\n{SEPARATOR}")
    print(f"🔍 모델: {model_id}")
    print(f"   경로: {path}")
    print(SEPARATOR)

    try:
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    except Exception as e:
        print(f"❌ 토크나이저 로드 실패: {e}")
        return

    # 1. 특수 토큰 정보
    print("\n📌 [특수 토큰]")
    print(f"  eos_token      : {repr(tokenizer.eos_token)} (id={tokenizer.eos_token_id})")
    print(f"  bos_token      : {repr(tokenizer.bos_token)} (id={tokenizer.bos_token_id})")
    print(f"  pad_token      : {repr(tokenizer.pad_token)} (id={tokenizer.pad_token_id})")
    print(f"  unk_token      : {repr(tokenizer.unk_token)} (id={tokenizer.unk_token_id})")

    # 2. additional_special_tokens
    print(f"\n📌 [additional_special_tokens]")
    for tok in tokenizer.additional_special_tokens:
        tid = tokenizer.convert_tokens_to_ids(tok)
        print(f"  {repr(tok):30s} -> id={tid}")

    # 3. chat_template 원문
    print(f"\n📌 [chat_template 원문]")
    if tokenizer.chat_template:
        # 너무 길면 앞부분만
        template = tokenizer.chat_template
        if len(template) > 800:
            print(template[:800])
            print(f"  ... (총 {len(template)}자, 생략됨)")
        else:
            print(template)
    else:
        print("  ⚠️  chat_template 없음!")

    # 4. 실제 포맷 결과
    print(f"\n📌 [apply_chat_template 결과]")
    try:
        formatted = tokenizer.apply_chat_template(
            TEST_MESSAGES,
            tokenize=False,
            add_generation_prompt=True
        )
        print(repr(formatted))
    except Exception as e:
        print(f"  ❌ 포맷 실패: {e}")
        return

    # 5. 토큰화 결과 (처음/끝 토큰 ID 확인)
    print(f"\n📌 [토큰 ID 분석]")
    try:
        token_ids = tokenizer.apply_chat_template(
            TEST_MESSAGES,
            tokenize=True,
            add_generation_prompt=True
        )
        print(f"  총 토큰 수: {len(token_ids)}")
        print(f"  처음 10개 ID: {token_ids[:10]}")
        print(f"  마지막 10개 ID: {token_ids[-10:]}")

        # 마지막 토큰들을 텍스트로 역변환
        last_tokens = tokenizer.convert_ids_to_tokens(token_ids[-10:])
        print(f"  마지막 10개 토큰: {last_tokens}")
    except Exception as e:
        print(f"  ❌ 토큰화 실패: {e}")

    # 6. stop token 후보 정리
    print(f"\n📌 [권장 stop_token_ids]")
    stop_ids = []
    if tokenizer.eos_token_id is not None:
        stop_ids.append((tokenizer.eos_token, tokenizer.eos_token_id))

    candidates = ["[|endofturn|]", "[|assistant|]", "[|user|]", "[|system|]",
                  "<|im_end|>", "<|endoftext|>", "<|EOT|>", "<|eot_id|>"]
    for tok in candidates:
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid != tokenizer.unk_token_id:
            stop_ids.append((tok, tid))

    for name, tid in stop_ids:
        print(f"  {repr(name):30s} -> id={tid}")

    print()


if __name__ == "__main__":
    print("🚀 모델별 Chat Template 비교 시작")
    for model_id in MODELS:
        analyze_model(model_id)

    print(f"\n{SEPARATOR}")
    print("✅ 완료!")
