import os
import sys
import torch
import wandb
import json
import weave
from tqdm import tqdm
from datasets import Dataset
from accelerate import Accelerator
from prompts import LIBRARY_QA_SYSTEM_PROMPT_COT_FEW_SHOT, LIBRARY_QA_USER_TEMPLATE

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from logger_config import setup_logger

logger = setup_logger()

class DataAugmentor :
    def __init__(self, model, tokenizer, config) :
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.accelerator = Accelerator()
        self.wandb_table = wandb.Table(columns=[
            "question", "answer", "label"
        ])

    # def check_imbalance(self, df):
    #     """클래스별 데이터 개수를 확인하고 부족한 클래스 리스트 반환"""
    #     counts = df['label'].value_counts()
    #     imbalanced_classes = counts[counts < self.config.MIN_SAMPLE_COUNT].index.tolist()
    #     return imbalanced_classes

    def _parse_output(self, raw_output):
        """JSON 파싱 로직"""
        try:
            if "```json" in raw_output:
                json_str = raw_output.split("```json")[1].split("```")[0].strip()
            elif "```" in raw_output:
                json_str = raw_output.split("```")[1].split("```")[0].strip()
            else:
                json_str = raw_output.strip()
            return json.loads(json_str)
        except Exception:
            return []
    
    @weave.op()
    def generate_samples(self, idx, faq_context) :
        user_content = LIBRARY_QA_USER_TEMPLATE.format(
            faq_content=faq_context
        )
        messages = [
            {"role": "system", "content": LIBRARY_QA_SYSTEM_PROMPT_COT_FEW_SHOT},
            {"role": "user", "content": user_content}
        ]

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.accelerator.device)

        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()

        with torch.no_grad() :
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.config.MAX_NEW_TOKENS,
                do_sample=True,
                temperature=self.config.TEMPERATURE,
                top_p=self.config.TOP_P,
                top_k=self.config.TOP_K,
                repetition_penalty=self.config.REPETITION_PENALTY,
                num_return_sequences=self.config.NUM_RETURN_SEQUENCES,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        prompt_len = input_ids.shape[-1]
        
        all_parsed_results = []

        for i in range(self.config.NUM_RETURN_SEQUENCES) :
            raw_text = self.tokenizer.decode(
                outputs[i][prompt_len:],
                skip_special_tokens=True # 
            )
            parsed_result = self._parse_output(raw_text)
            all_parsed_results.extend(parsed_result)
        
        return all_parsed_results

    def run_augmentation(self, dataset, output_file="augmented_data_simple.jsonl"):
        if self.accelerator.is_main_process:
            logger.info(f"🚀 데이터 증강 시작")
            with open(output_file, "w", encoding="utf-8") as f:
                pass

        total_count = 0
        disable_tqdm = not self.accelerator.is_main_process
        
        for idx, row in tqdm(enumerate(dataset), total=len(dataset), desc="Data Augmentation", disable=disable_tqdm):
            faq_context = row['DES']
            print(faq_context)
            
            try:
                result = self.generate_samples(idx, faq_context)

                if result and self.accelerator.is_main_process:
                    with open(output_file, "a", encoding="utf-8") as f:
                        for item in result:
                            item.update({
                                'faq': idx, 
                                'title': row.get('TITLE', ''),
                                'des': row.get('DES', '')
                            })
                            f.write(json.dumps(item, ensure_ascii=False) + "\n")
                            
                            # WandB 로깅
                            if len(result) > 0 and result.index(item) < 1:
                                self.wandb_table.add_data(
                                    item.get('question'), 
                                    item.get('answer'), 
                                    item.get('label')
                                )
                    total_count += len(result)

            except Exception as e:
                logger.error(f"⚠️ Error at index {idx}: {e}")
                continue
        
        if self.accelerator.is_main_process:
            logger.info(f"✅ 데이터 증강 완료! 생성된 샘플 수: {total_count}")
            logger.info(f"💾 저장 위치: {output_file}")
            
            # WandB 테이블 로깅
            if wandb.run:
                wandb.log({"generated_qa_samples": self.wandb_table})
                logger.info("📊 WandB에 생성된 샘플 테이블 로깅 완료")
        
        return output_file