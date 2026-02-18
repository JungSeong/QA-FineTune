import os
import sys
import torch
import wandb
import json
import weave
import pandas as pd
from tqdm import tqdm
from datasets import Dataset
from accelerate import Accelerator
from prompts import *

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from logger_config import get_infer_logger

logger = get_infer_logger()

class DataAugmentor :
    def __init__(self, model, tokenizer, config) :
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.accelerator = Accelerator()
        if self.tokenizer.pad_token is None :
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.wandb_table = wandb.Table(columns=["faq_id", "question", "answer", "label"])
        self.labels = ["yes", "no", "info", "false"]
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

    def preprocess_data(self, file_path="../data/json/augmented_data.jsonl") :
        if not self.accelerator.is_main_process :
            return None
        if not os.path.exists(file_path) :
            logger.error(f"⚠️ 파일이 존재하지 않습니다: {file_path}")
            return None
        
        logger.info("🔍 데이터 불균형 분석 및 정제 시작...")
        try :
            df = pd.read_json(file_path, lines=True)
            initial_count = len(df)
            logger.info(f"📂 초기 데이터 수: {initial_count}")

            required_cols = ["faq_id","question", "answer", "label"]
            
            # 1. 필수 칼럼이 없을 경우, 우선 필수 컬럼을 None으로라도 채움
            for col in required_cols :
                if col not in df.columns :
                    df[col] = None

            # 2. 결측치 제거
            nan_rows = df[df[required_cols].isna().any(axis=1)]
            if not nan_rows.empty:
                logger.warning(f"⚠️ 결측치로 제거되는 데이터 (총 {len(nan_rows)}건):")
                # 로그에 전체 행 내용 출력 (최대 20건)
                logger.warning(nan_rows.head(20).to_json(orient='records', force_ascii=False, indent=2))
                if len(nan_rows) > 20:
                    logger.warning(f"...외 {len(nan_rows)-20}건 생략")

            df_clean = df.dropna(subset=required_cols)

            # 3. 빈 문자열("") 및 공백만 있는 데이터 제거
            for col in required_cols :
                df_clean[col] = df_clean[col].astype(str).str.strip()
                empty_rows = df_clean[df_clean[col] == ""]
                if not empty_rows.empty:
                    logger.warning(f"⚠️ '{col}' 컬럼이 비어있어 제거되는 데이터 (총 {len(empty_rows)}건):")
                    logger.warning(empty_rows.head(20).to_json(orient='records', force_ascii=False, indent=2))
                df_clean = df_clean[df_clean[col] != ""]

            # 4. 중복 데이터 제거
            duplicates = df_clean[df_clean.duplicated(subset=required_cols, keep='first')]
            if not duplicates.empty:
                logger.warning(f"⚠️ 중복되어 제거되는 데이터 (총 {len(duplicates)}건):")
                logger.warning(duplicates.head(20).to_json(orient='records', force_ascii=False, indent=2))

            df_clean = df_clean.drop_duplicates(subset=required_cols, keep="first")

            # 5. 유효하지 않은 라벨 제거
            allowed_labels = self.labels
            invalid_label_rows = df_clean[~df_clean['label'].isin(allowed_labels)]
            
            if not invalid_label_rows.empty:
                logger.warning(f"⚠️ 유효하지 않은 라벨(Label)로 제거되는 데이터 (총 {len(invalid_label_rows)}건):")
                logger.warning(f"👉 허용된 라벨: {allowed_labels}")
                logger.warning(f"👉 발견된 이상 라벨 예시: {invalid_label_rows['label'].unique().tolist()[:10]}") # 어떤 이상한 값이 있는지 확인용
                logger.warning(invalid_label_rows.head(20).to_json(orient='records', force_ascii=False, indent=2))
                
                if len(invalid_label_rows) > 20:
                    logger.warning(f"...외 {len(invalid_label_rows)-20}건 생략")

            # 6. 인덱스 초기화
            df_clean = df_clean.reset_index(drop=True)

            final_count = len(df_clean)
            dropped_count = initial_count - final_count

            if dropped_count > 0:
                logger.warning(f"🧹 정제 완료: {dropped_count}개 불량 데이터 제거됨 (최종: {final_count}개)")
                df_clean.to_json(file_path, orient='records', lines=True, force_ascii=False)
            else:
                logger.info("✨ 데이터가 이미 깨끗합니다.")

            return df_clean

        except Exception as e :
            logger.error(f"⚠️ 데이터 불균형 분석 및 정제 중 오류 발생: {e}")

    def find_imbalanced_tasks(self, original_dataset, file_path="../data/json/augmented_data.jsonl"):
        # 1. Main process runs preprocessing (cleaning and saving to file)
        if self.accelerator.is_main_process:
            cleaned_data = self.preprocess_data(file_path)
        
        # Wait for main process to finish writing
        self.accelerator.wait_for_everyone()

        # 2. All processes read the data to calculate counts consistently
        counts = pd.DataFrame()

        if cleaned_data is not None :
            try :
                df = cleaned_data
                if not df.empty and 'faq_id' in df.columns and 'label' in df.columns:
                    counts = df.groupby(['faq_id', 'label']).size().unstack(fill_value=0)
                    # Only main process prints the table to avoid clutter
                    if self.accelerator.is_main_process:
                        logger.info(f"📊 현재 데이터 분포:\n{counts}")
            except Exception as e :
                logger.error(f"⚠️ 데이터를 읽어올 수 없습니다: {e}")

        missing_tasks = []
        required_labels = self.labels

        for idx, row in enumerate(original_dataset) :
            faq_context = row["DES"]
            needed_labels = []

            if idx in counts.index :
                current_counts = counts.loc[idx]
                for label in self.labels:
                    if label not in current_counts or current_counts[label] < 0:
                        needed_labels.append(label)
            else:
                # 데이터가 하나도 없는 경우 (삭제됨 or 누락됨) -> 전부 생성
                needed_labels = self.labels

            if needed_labels:
                missing_tasks.append({
                    "idx": idx,              # 원본 데이터 인덱스
                    "context": faq_context,  # 원본 텍스트
                    "targets": needed_labels,# 필요한 라벨들
                    "meta_title": row.get('TITLE', ''),
                    "meta_des": row.get('DES', '')
                })
        
        logger.info(f"missing_tasks: {missing_tasks}")
        return missing_tasks

    def _run_model_generate(self, messages, num_return_sequences) :
        model_inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.accelerator.device)
        
        input_ids = model_inputs['input_ids']
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()
        prompt_len = input_ids.shape[-1]

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
        
        all_parsed_results = []

        for i in range(self.config.NUM_RETURN_SEQUENCES) :
            raw_text = self.tokenizer.decode(
                outputs[i][prompt_len:],
                skip_special_tokens=True # 
            )
            parsed_result = self._parse_output(raw_text)
            all_parsed_results.extend(parsed_result)
        
        return all_parsed_results

    @weave.op()
    def generate_samples(self, idx, faq_context) :
        user_content = LIBRARY_QA_USER_TEMPLATE.replace("{faq_content}", faq_context)
        messages = [
            {"role": "system", "content": LIBRARY_QA_SYSTEM_PROMPT_COT_FEW_SHOT},
            {"role": "user", "content": user_content}
        ]
        logger.info(messages)
        return self._run_model_generate(messages, self.config.NUM_RETURN_SEQUENCES)
    
    @weave.op()
    def generate_targeted_samples(self, idx, faq_context, target_labels) :
        target_labels_str = ", ".join(target_labels)
        all_guidelines = ALL_GUIDELINES
        selected_guidelines = "\n".join([all_guidelines[l] for l in target_labels if l in all_guidelines])
        selected_examples = ",\n      ".join([ALL_COT_FEW_SHOT_EXAMPLES[l] for l in target_labels if l in ALL_COT_FEW_SHOT_EXAMPLES])

        system_content = LIBRARY_QA_TARGETED_SYSTEM_PROMPT_COT_FEW_SHOT.replace("{target_labels}", target_labels_str).replace("{selected_guidelines}", selected_guidelines).replace("{selected_examples}", selected_examples)
        user_content = LIBRARY_QA_USER_TEMPLATE.replace("{faq_content}", faq_context)
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]
        logger.info(messages)
        return self._run_model_generate(messages, self.config.NUM_RETURN_TARGET_SEQUENCES)

    def _run_generation_loop(self, items, output_file, mode="initial") :
        if not items :
            return
        desc = "Initial Gen" if mode == "initial" else "Targeted Gen"
        disable_tqdm = not self.accelerator.is_main_process

        base, ext = os.path.splitext(output_file)
        rank_output_file = f"{base}_rank{self.accelerator.process_index}{ext}"

        with open(rank_output_file, "a", encoding="utf-8") as f:
            for item in tqdm(items, total=len(items), desc=f"{desc} (Rank {self.accelerator.process_index})", disable=disable_tqdm):
                try :
                    if mode == "initial" :
                        idx, row = item
                        faq_context = row['DES']
                        result = self.generate_samples(idx, faq_context)
                        meta = {'faq_id': idx, 'original_title': row.get('TITLE', '')}
                    else: # targeted
                        idx = item['idx']
                        faq_context = item['context']
                        targets = item['targets']
                        meta = {'faq_id': idx, 'original_title': item['meta_title']}
                        result = self.generate_targeted_samples(idx, faq_context, targets)
                    
                    logger.info(result)
                    logger.info(meta)
                    # 저장
                    if result:
                        for res in result:
                            # 🌟 [핵심] FAQ ID를 데이터에 주입 (나중에 groupby용)
                            res.update(meta)
                            f.write(json.dumps(res, ensure_ascii=False) + "\n")

                            # WandB 로깅 (Main Process만, 첫 번째 샘플만)
                            if self.accelerator.is_main_process and len(result) > 0 and result.index(res) == 0:
                                self.wandb_table.add_data(
                                    res.get('faq_id'), res.get('question'), res.get('answer'), res.get('label')
                                    )
                        f.flush()
                        os.fsync(f.fileno())

                except Exception as e:
                    import traceback
                    logger.error(f"💥 치명적 에러 발생 (idx: {idx if 'idx' in locals() else '?'}): {e}")
                    logger.error(traceback.format_exc()) # <-- 이게 범인을 알려줍니다.
                    
                    # 디버깅을 위해 여기서 멈추게 하려면 raise를 쓰세요
                    raise e

    def merge_rank_files(self, output_file) :
        if not self.accelerator.is_main_process:
            return

        logger.info("📦 분산된 데이터 파일 병합 중...")
        base, ext = os.path.splitext(output_file)

        with open(output_file, "a", encoding="utf-8") as outfile:
            for rank in range(self.accelerator.num_processes):
                rank_file = f"{base}_rank{rank}{ext}"
                if os.path.exists(rank_file):
                    with open(rank_file, "r", encoding="utf-8") as infile:
                        # 파일 내용 복사
                        import shutil
                        shutil.copyfileobj(infile, outfile)
                    
                    try :
                        logger.info(f"병합 후 조각 파일 삭제 (중복 병합 방지): {rank_file}")
                        os.remove(rank_file) 
                    except OSError as e:
                        logger.error(f"💥 치명적 에러 발생 (idx: {idx if 'idx' in locals() else '?'}): {e}")
                        logger.error(traceback.format_exc()) # <-- 이게 범인을 알려줍니다.
                        
        logger.info(f"✅ 병합 완료: {output_file}")
    
    def run_pipeline(self, dataset, output_file="../data/json/augmented_data.jsonl"):
        # --- 1. 초기 생성 (Initial Pass) ---
        if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
            if self.accelerator.is_main_process:
                logger.info("🚀 [Step 1] 초기 데이터 생성 시작...")
                with open(output_file, "w", encoding="utf-8") as f: pass
            
            # 데이터셋에 인덱스 부여해서 리스트로 변환 (idx, row)
            indexed_dataset = list(enumerate(dataset))
            
            # 멀티 GPU 분산 (Sharding)
            my_items = indexed_dataset[self.accelerator.process_index::self.accelerator.num_processes]
            self._run_generation_loop(my_items, output_file, mode="initial")
        else:
            logger.info("📂 기존 파일이 이미 존재합니다. 데이터를 검수합니다.")

        self.accelerator.wait_for_everyone()
        self.merge_rank_files(output_file)

        # --- 2. 반복 보완 (Iterative Refinement) ---
        max_iter = getattr(self.config, 'MAX_AUG_ITERATIONS', 3)
        
        for i in range(1, max_iter + 1):
            self.accelerator.wait_for_everyone() # 동기화
            
            # 정제 및 부족분 분석 (내부적으로 동기화 포함됨)
            missing_tasks = self.find_imbalanced_tasks(dataset, output_file)
            logger.info(missing_tasks)
            
            if not missing_tasks:
                if self.accelerator.is_main_process:
                    logger.info("✨ 모든 데이터 균형 완료! 증강 종료.")
                break
            
            if self.accelerator.is_main_process:
                logger.info(f"📉 [Step 2-{i}] 부족한 작업: {len(missing_tasks)}건. 추가 생성...")

            # 부족분 분산 처리 (Sharding)
            my_tasks = missing_tasks[self.accelerator.process_index::self.accelerator.num_processes]
            
            if len(my_tasks) > 0:
                self._run_generation_loop(my_tasks, output_file, mode="targeted")
            
            self.accelerator.wait_for_everyone()
            self.merge_rank_files(output_file)

        # 완료 로그
        if self.accelerator.is_main_process:
            if wandb.run:
                wandb.log({"generated_qa_samples": self.wandb_table})
                logger.info("📊 WandB 업로드 완료")

        return output_file