from transformers import AutoTokenizer
from config import Config

def generate_prompts(examples):
    config = Config()
    """배치 처리를 위한 함수"""
    question = examples["question"]
    answer = examples["answer"]
    context = examples["original_title"]
    
    system_message = (
        "당신은 도서관 운영에 대한 전문적인 지식을 가진 인공지능 사서입니다. "
        "제공된 [도서관 정보]를 바탕으로 사용자의 질문에 정확하고 친절하게 답하십시오. "
        "정보에 없는 내용은 함부로 추측하지 말고 정중히 확인이 어렵다고 답하세요."
    )

    user_content = (
        f"### [도서관 정보]\\n{context}\\n\\n"
        f"### [질문]\\n{question}\\n\\n"
        f"### [지시 사항]\\n"
        f"1. 친절한 말투로 규정에 근거하여 답변할 것.\\n"
        f"2. 3문단 이내로 답변할 것.\\n"
        f"3. 답변 끝에 지시 사항을 반복하지 말 것."
    )

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": answer}
    ]

    tokenizer = AutoTokenizer.from_pretrained(config.LOCAL_MODEL_DIR, trust_remote_code=True)

    # apply_chat_template 적용
    full_prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=False
    )
    
    return full_prompt