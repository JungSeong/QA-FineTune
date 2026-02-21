import os
import sys
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from config import Config

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from train.preprocess_dataset import preprocess_dataset
from logger_config import get_eval_logger

logger = get_eval_logger()

config = Config()

def load_LoRA_model():
    base_model_id = Config.MODEL_ID
    local_model_dir = Config.LOCAL_MODEL_DIR
    adapter_path = f"{Config.ADAPTER_PATH}/final"

    model = AutoModelForCausalLM.from_pretrained(
        config.LOCAL_MODEL_DIR,
        quantization_config=config.QUANTIZATION_CONFIG,
        trust_remote_code=True,
        device_map="auto", # Accelerate가 자동 관리하도록 설정
    )

    if hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
        model.get_input_embeddings = lambda: model.transformer.wte

    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    
    model.eval()

    return model, tokenizer

def generate_answer(model, tokenizer, question, context) :
    system_message = (
        "당신은 도서관 운영에 대한 전문적인 지식을 가진 인공지능 사서입니다. "
        "제공된 [도서관 정보]를 바탕으로 사용자의 질문에 정확하고 친절하게 답하십시오. "
        "정보에 없는 내용은 함부로 추측하지 말고 정중히 확인이 어렵다고 답하세요."
    )

    user_content = (
        f"### [도서관 정보]\n{context}\n\n"
        f"### [질문]\n{question}\n\n"
        f"### [지시 사항]\n"
        f"1. 친절한 말투로 규정에 근거하여 답변할 것.\n"
        f"2. 3문단 이내로 답변할 것.\n"
        f"3. 답변 끝에 지시 사항을 반복하지 말 것."
    )

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_content}
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True, trust_remote_code=True)
    return response

def run_benchmark():
    model, tokenizer = load_LoRA_model()
    results = []

    logger.info(f"🧐 학습된 {Config.MODEL_ID}/final 모델로 추론 시작...")
    dataset = preprocess_dataset()
    test_dataset = dataset['test']

    for row in tqdm(test_dataset):
        question = row['question']
        context = row['original_title']
        ground_truth = row['answer']
        
        # 모델의 답변 생성
        model_generated = generate_answer(model, tokenizer, question, context)
        logger.info(f"Question: {question}")
        logger.info(f"Ground Truth: {ground_truth}")
        logger.info(f"Model Generated: {model_generated}")
        
        results.append({
            "question": question,
            "ground_truth": ground_truth,
            "model_generated": model_generated
        })

    # 결과를 CSV로 저장 (나중에 판사 모델에게 전달용)
    result_df = pd.DataFrame(results)
    result_df.to_csv(f"{Config.BENCHMARK_PATH}/benchmark_results_{Config.MODEL_ID}.csv", index=False)
    return result_df

# 실행 부분
if __name__ == "__main__":
    run_benchmark()