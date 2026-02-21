import os
import sys
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from logger_config import get_infer_logger
logger = get_infer_logger()

class HeavyModelInference:
    def __init__(self, model_path: str):
        self.model_path = model_path
        
        # 1. 토크나이저 로드 및 학습 시 설정 이식
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # [🌟 요청 사항 반영 1] 패딩 토큰 설정
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info(f"✅ Pad token set to EOS: {self.tokenizer.pad_token}")

        # 2. 모델 로드 (Accelerate device_map="auto" 사용)
        logger.info(f"🔄 모델 로딩 시작: {model_path} (BF16)")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,    # 32B 정밀도 유지
            device_map="auto",             # 🌟 여러 GPU에 자동 분산 (Accelerate 핵심)
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )

        # [🌟 요청 사항 반영 2] 임베딩 레이어 명시적 연결 (EXAONE 구조 대응)
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "wte"):
            self.model.get_input_embeddings = lambda: self.model.transformer.wte
            logger.info("✅ Input embeddings mapping applied (transformer.wte).")

        self.model.eval()
        self._debug_tokens()

    def _debug_tokens(self):
        """Suspicious ID 361 검증을 위한 디버깅 코드"""
        tokens_to_check = ["[|endofturn|]", "[|assistant|]", "[|user|]", "[|system|]"]
        logger.info("--- [Token ID Verification] ---")
        for t in tokens_to_check:
            tid = self.tokenizer.convert_tokens_to_ids(t)
            logger.info(f"Token: {t:15} | ID: {tid}")
        logger.info(f"EOS ID: {self.tokenizer.eos_token_id} | PAD ID: {self.tokenizer.pad_token_id}")
        logger.info("-------------------------------")

    @torch.no_grad()
    def generate_text(self, messages: List[Dict[str, str]], max_tokens=512):
        # 1. Chat Template 적용
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # 2. 토큰화 및 장치 이동
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # 3. 생성 설정 (vLLM 파라미터와 1:1 대응)
        generation_params = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "max_new_tokens": max_tokens,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.2,     # 🌟 루프 방지 강화
            "do_sample": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.convert_tokens_to_ids("[|endofturn|]")
        }

        # 4. 추론 실행
        outputs = self.model.generate(**generation_params)
        
        # 5. 후처리 (Prompt 부분 제외하고 Decode)
        new_ids = outputs[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True)

# 실행 예시
if __name__ == "__main__":
    MODEL_PATH = "/home/vsc/LLM/model/Exaone-3.5-32B-Instruct"
    infer = HeavyModelInference(MODEL_PATH)
    
    sample_msgs = [
        {"role": "system", "content": "너는 도서관 안내 전문가야."},
        {"role": "user", "content": "대화도서관 휴관일이 언제야?"}
    ]
    
    result = infer.generate_text(sample_msgs)
    print(f"\n[Final Output]\n{result}")