import torch
import gc
import re
import os
import sys
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict
from logger_config import get_infer_logger
from datetime import datetime

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from logger_config import get_infer_logger
logger = get_infer_logger()

class LLMEvaluator:
    def __init__(self, model_path: str, log_path: str):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.log_path = log_path

    def _write_log(self, data: Dict):
        with open(self.log_path, mode='a', encoding='utf-8') as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    def load(self):
        """평가 시점에만 VRAM에 로드"""
        if self.model is not None: return
        logger.info(f"⚖️ [Eval] 로컬 평가 모델 로딩: {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # 🌟 학습 때 사용했던 필수 설정 적용
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "wte"):
            self.model.get_input_embeddings = lambda: self.model.transformer.wte
        self.model.eval()

    def unload(self):
        """메모리 반환 (vLLM을 위해)"""
        if self.model is not None:
            del self.model
            del self.tokenizer
            gc.collect()
            torch.cuda.empty_cache()
            self.model = None
            logger.info("✅ [Eval] 로컬 모델 언로드 완료")

    @torch.no_grad()
    def evaluate_batch(self, data_list: List[Dict], criteria: str) -> List[Dict]:
        """로컬 모델로 정석 채점 수행"""
        results = []
        for item in data_list:
            prompt_text = f"""
                당신은 도서관 QA 데이터를 바탕으로 QA 쌍의 품질을 평가하는 전문가입니다.
                주어진 데이터를 분석한 후, **추론 과정**을 꼭 순서대로 따라가며 생각하세요.

                [추론 과정]
                기준 이해 및 키워드 정리: 주어진 criteria는 "질문과 답변이 논리적으로 일관되는가"이며, 특히 답변과 {item['label']} 과의 유사성을 판단.
                label 유형('yes','no': 가/불가를 묻는 질문 'info': 가/불가 아닌 정보 질문, 'false': 모호한 질문)을 기준으로 QA 쌍의 적합성 평가.
                키워드: 논리 일관성(질문-답변 매칭), label 적합성(유형 일치 여부)
                평가 항목 분해: 품질을 1~10점으로 평가. 0~1점(매우 낮음: 논리 불일치 또는 label 완전 불일치), 2~3점(낮음: 부분 불일치), 4~6점(보통: 기본 일치하나 약점 있음), 7~8점(높음: 대부분 일치), 9~10점(매우 높음: 완벽 일치).
                논리 일관성: 질문이 요구하는 내용에 답변이 정확히 맞는지(예: 가부 질문에 info 답변 하지 않음).
                label 유사성: label이 'yes'라면 질문이 가부형이고 답변이 긍정적·일치적, 'no'라면 부정적·어긋남, 'info'라면 설명형, 'false'라면 판단 불가나 오류.

                QA 쌍 분석:
                질문 유형 분류(가부/정보/모호 등).
                답변 내용 검토: 질문에 논리적으로 맞는지, 자연스럽고 정확한지. 
                label 매칭: item['label']과 QA 쌍의 실제 성격이 일치하는지(예: label 'yes'인데 답변이 부정적이면 불일치).

                점수 산정 기준 적용:
                논리 일관성 만점: 3점 배분(질문-답변 매칭 완벽 시).
                label 유사성 만점: 7점 배분(label 완벽 일치 시).

                최종 출력 예시 (JSON):
                {{"score": 8, "reason": "질문과 답변이 논리적으로 완벽히 일관되며, label 'no'와의 유사성도 우수함."}}
                {{"score": 3, "reason": "질문과 답변이 논리적으로 일관되지 못하며, 특히 정보성 질문에 yes로 답변함"}}
            """
            # apply_chat_template 혹은 f-string 사용
            inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1, # 일관성
                repetition_penalty=1.15,
                eos_token_id=361 # [|endofturn|]
            )
            
            raw_res = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
            logger.info(raw_res)
            # 간단 파싱
            score = 3
            try:
                # JSON 형태 추출 시도
                json_match = re.search(r'\{.*\}', raw_res, re.DOTALL)
                if json_match:
                    score = int(json.loads(json_match.group()).get("score", 3))
                else:
                    # 숫자가 맨 앞에 나오는 경우 (로그 관찰 결과 대응)
                    digit_match = re.search(r'^\d', raw_res)
                    if digit_match: score = int(digit_match.group())
            except:
                pass
            
            eval_entry = {
                **item,
                "eval_score": score,
                "eval_raw_output": raw_res,
                "criteria": criteria
            }
            
            # 🌟 즉시 로그 기록
            self._write_log(eval_entry)
            results.append(eval_entry)
        return results