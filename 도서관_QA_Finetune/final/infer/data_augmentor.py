import os
import asyncio
import json
import sys
import pandas as pd
import aiofiles
import wandb
import weave
import re
import time
import numpy as np
from datetime import datetime
from tqdm.asyncio import tqdm
from transformers import AutoTokenizer
from mcp import ClientSession
from mcp.client.sse import sse_client
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from kiwipiepy import Kiwi
from prompts import *
from evaluator import LLMEvaluator

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from logger_config import get_infer_logger
logger = get_infer_logger()

class AsyncDataAugmentor:
    def __init__(self, mcp_url, model_id, config):
        self.mcp_url = mcp_url
        self.model_id = model_id
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.semaphore = asyncio.Semaphore(getattr(config, 'NUM_SEMAPHORES', 3))
        self.labels = config.LABELS
        self.write_queue = asyncio.Queue()
        self.output_file = getattr(config, 'AUGMENTED_DATA_PATH', "augmented_data.jsonl")
        self.kiwi = Kiwi() # 형태소 분석기 초기화
        self.perf_stats = {
            "total_tokens": 0,
            "total_time": 0.0,
            "total_samples": 0
        }
        self.evaluator = LLMEvaluator(config.EVAL_MODEL_PATH, config.EVAL_LOG_PATH)

    # 1. 🌟 강력해진 JSON 파서 (여러 객체 및 마크다운 완벽 대응)
    def _parse_output(self, raw_output):
        # 마크다운 코드 블록 제거
        clean_text = re.sub(r'```json\s*|```', '', raw_output).strip()
        
        # 정규표현식으로 { ... } 패턴을 모두 추출 (가장 확실한 방법)
        json_pattern = re.compile(r'\{.*?\}', re.DOTALL)
        matches = json_pattern.findall(clean_text)
        
        results = []
        for match in matches:
            try:
                # 개행 문자 등으로 깨진 JSON 수정 후 로드
                item = json.loads(match.replace('\n', ' '))
                results.append(item)
            except json.JSONDecodeError:
                continue
        return results

    def calculate_gini(self, counts):
        """지니 계수 계산: 0(완전 균형) ~ 1(완전 불균형)"""
        counts = np.array(counts, dtype=np.float64)
        if np.sum(counts) == 0: return 0.0
        n = len(counts)
        sorted_counts = np.sort(counts)
        index = np.arange(1, n + 1)
        return (np.sum((2 * index - n - 1) * sorted_counts)) / (n * np.sum(sorted_counts))

    def calculate_self_bleu(self, df, sample_size=100):
        if len(df) < 2: return 0.0
        sample_texts = df['answer'].sample(min(len(df), sample_size)).tolist()
        
        tokenized = []
        for text in sample_texts:
            tokens = [t.form for t in self.kiwi.tokenize(text)]
            tokenized.append(tokens)
            
        scores = []
        smooth = SmoothingFunction().method1
        for i in range(len(tokenized)):
            ref = tokenized[:i] + tokenized[i+1:]
            hypo = tokenized[i]
            score = sentence_bleu(ref, hypo, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)
            scores.append(score)
        return np.mean(scores) if scores else 0.0

    def calculate_metrics(self, df):
        """데이터프레임 하나만 받아서 내부에서 모든 지표를 계산합니다."""
        if len(df) == 0:
            return 0.0, 0.0
        # 1. 라벨 빈도수를 내부에서 직접 추출 (reindex로 모든 라벨 포함)
        counts = df['label'].value_counts().reindex(self.labels, fill_value=0).tolist()
        # 2. 지니 계수 계산
        gini_val = self.calculate_gini(counts)
        # 3. Self-BLEU 계산
        self_bleu_val = self.calculate_self_bleu(df)
        
        return gini_val, self_bleu_val

    def find_imbalanced_tasks(self, file_path, original_dataset):
        logger.info(f"📊 데이터 품질 및 분포 분석 시작: {file_path}")
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            df = pd.DataFrame(columns=["faq_id", "label", "answer"])
        else:
            df = pd.read_json(file_path, lines=True)

        df = df.drop_duplicates(subset=["faq_id", "question", "answer", "label"])
        total_count = len(df)
        
        # 1. 기본 라벨 통계
        label_stats = pd.DataFrame(index=self.labels)
        if total_count > 0:
            counts = df['label'].value_counts()
            label_stats['count'] = label_stats.index.map(lambda x: counts.get(x, 0))
            label_stats['percentage'] = (label_stats['count'] / total_count * 100).round(2)
        else:
            label_stats['count'], label_stats['percentage'] = 0, 0.0

        gini_val, self_bleu_val = self.calculate_metrics(df)

        # 3. 📝 로그 출력 (더 상세하게)
        logger.info(f"\n{'═'*50}\n"
                    f"📈 데이터셋 리포트 (총 {total_count}개)\n"
                    f"{'-'*50}\n"
                    f"{label_stats.to_string()}\n"
                    f"{'-'*50}\n"
                    f"⚖️ 불균형 지수 (Gini): {gini_val:.3f} (0에 가까울수록 좋음)\n"
                    f"🧩 다양성 지수 (KoBLEU): {self_bleu_val:.3f} (낮을수록 좋음)\n"
                    f"{'═'*50}")

        # 4. WandB 로깅
        if wandb.run:
            wandb.log({
                "quality/gini": gini_val,
                "quality/self_bleu": self_bleu_val,
                "quality/total_count": total_count,
                **{f"dist/{l}": c for l, c in label_stats['count'].items()}
            })

        # 5. 부족한 태스크 추출 로직
        dist = df.groupby(['faq_id', 'label']).size().unstack(fill_value=0)
        for l in self.labels:
            if l not in dist.columns: dist[l] = 0

        missing_tasks = []
        for idx, row in enumerate(original_dataset):
            needed = [l for l in self.labels if idx not in dist.index or dist.loc[idx, l] < 1]
            if needed:
                missing_tasks.append({
                    "idx": idx,
                    "context": row.get("DES", ""),
                    "targets": needed,
                    "meta_title": row.get("TITLE", "")
                })
                
        # 나중에 노션에 기록할 때 쓸 통계 데이터를 함께 반환하면 좋습니다.
        return missing_tasks, {"gini": gini_val, "self_bleu": self_bleu_val, "count": total_count}

    async def _save_worker(self):
        logger.info(f"💾 실시간 저장 워커 시작: {self.output_file}")
        async with aiofiles.open(self.output_file, mode='a', encoding='utf-8') as f:
            while True:
                result = await self.write_queue.get()
                if result is None: break
                
                items = result if isinstance(result, list) else [result]
                for item in items:
                    await f.write(json.dumps(item, ensure_ascii=False) + "\n")
                
                await f.flush()
                self.write_queue.task_done()

    @weave.op()
    async def _mcp_generate_single(self, session, messages, meta):
        async with self.semaphore:
            start_t = time.perf_counter()
            try:
                response = await session.call_tool("generate_text", {
                    "messages": messages,
                    "max_tokens": getattr(self.config, 'MAX_NEW_TOKENS', 1024),
                    "temperature": getattr(self.config, 'TEMPERATURE', 0.7),
                    "top_p": getattr(self.config, 'TOP_P', 0.95),
                    "top_k": getattr(self.config, 'TOP_K', 20)
                })
                raw_text = response.content[0].text
                end_t = time.perf_counter()

                duration = end_t - start_t
                parsed_results = self._parse_output(raw_text)

                logger.info(parsed_results)
                
                for res in parsed_results:
                    res.update(meta)
                
                if parsed_results:
                    for res in parsed_results:
                        token_count = len(self.tokenizer.encode(res.get('answer', '')))
                        self.perf_stats["total_tokens"] += token_count
                        self.perf_stats["total_samples"] += 1
                        res.update(meta)
                    
                    self.perf_stats["total_time"] += duration
                    await self.write_queue.put(parsed_results)
                return parsed_results
            except Exception as e:
                logger.warning(f"⚠️ 추론 중 에러 발생: {e}")
                return []

    def print_epoch_report(self, iteration_name, quality_stats, perf_stats):
        """매 회차 종료 시 출력될 통합 리포트"""
        logger.info(f"\n{'#'*60}\n"
                    f"📢 [{iteration_name}] 단계 완료 리포트\n"
                    f"{'-'*60}\n"
                    f"📊 [품질] Gini 지수: {quality_stats['gini']:.3f} | KoBLEU: {quality_stats['self_bleu']:.3f}\n"
                    f"⚡ [성능] TPS: {perf_stats['tps']:.2f} | Sec/Sample: {perf_stats['sec_per_sample']:.2f}s\n"
                    f"📈 [누적] 총 샘플 수: {quality_stats['count']}개\n"
                    f"{'#'*60}")
        
        # WandB에 단계별 기록
        if wandb.run:
            wandb.log({
                "epoch": iteration_name,
                "metrics/gini": quality_stats['gini'],
                "metrics/ko_bleu": quality_stats['self_bleu'],
                "perf/tps": perf_stats['tps'],
                "perf/sec_per_sample": perf_stats['sec_per_sample']
            })

    def get_final_report(self):
        tps = self.perf_stats["total_tokens"] / self.perf_stats["total_time"] if self.perf_stats["total_time"] > 0 else 0
        sec_per_sample = self.perf_stats["total_time"] / self.perf_stats["total_samples"] if self.perf_stats["total_samples"] > 0 else 0
        
        report = (
            f"\n{'='*50}\n"
            f"⚡ 실시간 생성 성능 리포트\n"
            f"{'-'*50}\n"
            f"🚀 TPS (Tokens/Sec): {tps:.2f}\n"
            f"⏱️ Sec/Sample: {sec_per_sample:.2f}s\n"
            f"🔢 총 생성 토큰: {self.perf_stats['total_tokens']}\n"
            f"📦 총 생성 샘플: {self.perf_stats['total_samples']}\n"
            f"{'='*50}"
        )
        logger.info(report)
        return {"tps": tps, "sec_per_sample": sec_per_sample}

    async def run_generation_batch(self, session, tasks, mode="initial"):
        async_tasks = []
        for item in tasks:
            if mode == "initial":
                idx, row = item
                faq_context = row['DES']
                meta = {'faq_id': idx, 'original_title': row.get('TITLE', ''), 'iteration': mode}
                user_content = LIBRARY_QA_USER_TEMPLATE.replace("{faq_content}", faq_context)
                messages = [
                    {"role": "system", "content": LIBRARY_QA_SYSTEM_PROMPT_NO_COT},
                    {"role": "user", "content": user_content}
                ]
            else:
                # 🌟 Targeted 모드 구조 수정 (TypeError 방지)
                idx = item['idx']
                faq_context = item['context']
                target_labels = item['targets']
                meta = {'faq_id': idx, 'original_title': item['meta_title'], 'iteration': mode}

                logger.info(f"부족한 Task : {target_labels}")
                
                target_labels_str = ", ".join(target_labels)
                # 프롬프트 조립 로직 생략(기존 유지)
                system_content = LIBRARY_QA_TARGETED_SYSTEM_PROMPT_NO_COT.replace("{target_labels}", target_labels_str)
                user_content = LIBRARY_QA_USER_TEMPLATE.replace("{faq_content}", faq_context)
                messages = [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content}
                ]
            
            async_tasks.append(self._mcp_generate_single(session, messages, meta))

        results = await tqdm.gather(*async_tasks, desc=f"🚀 {mode} 증강 진행 중")
        return [res for sublist in results for res in sublist]

    async def run_pipeline_async(self, dataset, output_file):
        async with sse_client(self.mcp_url) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                await session.call_tool("switch_model", {
                    "model_id": self.config.GEN_SERVER_MODEL_NAME,
                    "config": {"trust_remote_code": True, "gpu_memory_utilization": 0.7}
                })

                save_worker_task = asyncio.create_task(self._save_worker())

                try:
                    # --- [STEP 1] 초기 생성 (Epoch 0) ---
                    if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
                        logger.info("🚀 [Initial] 초기 데이터 생성 시작...")
                        await self.run_generation_batch(session, list(enumerate(dataset)), mode="initial")
                        
                        # 초기 생성 후 리포트
                        missing, q_stats = self.find_imbalanced_tasks(output_file, dataset)
                        p_stats = self.get_final_report() # 현재까지의 성능 계산
                        self.print_epoch_report("Initial", q_stats, p_stats)
                    else:
                        # 파일에서 데이터 로드 (new_data 할당 에러 방지)
                        new_data = []
                        async with aiofiles.open(output_file, mode='r', encoding='utf-8') as f:
                            async for line in f:
                                if line.strip(): new_data.append(json.loads(line))

                    logger.info("🧹 [MCP] VRAM 확보를 위해 모델 언로드 요청...")
                    await session.call_tool("unload_model")
                    
                    self.evaluator.load()
                    try:
                        logger.info("⚖️ [Eval] 로컬 모델 기반 품질 검증 실행")
                        # 동기 함수이므로 to_thread로 실행
                        validated_data = await asyncio.to_thread(
                            self.evaluator.evaluate_batch, 
                            new_data[:20], 
                            "질문과 답변이 논리적으로 일관되는가?"
                        )
                        avg_score = np.mean([d['eval_score'] for d in validated_data])
                        logger.info(f"⭐ [Initial] LLM Judge 평균 점수: {avg_score:.2f} / 5.0")
                    finally:
                        self.evaluator.unload()

                    # --- [STEP 2] 반복 보완 (Epoch 1, 2, ...) ---
                    max_iter = getattr(self.config, 'MAX_AUG_ITERATIONS', 2)
                    for i in range(1, max_iter + 1):
                        missing_tasks, _ = self.find_imbalanced_tasks(output_file, dataset)
                        if not missing_tasks:
                            logger.info("✅ 모든 라벨이 균형을 이뤘습니다. 보완 종료.")
                            break
                        
                        await session.call_tool("switch_model", {
                            "model_id": self.config.GEN_SERVER_MODEL_NAME,
                            "config": {"trust_remote_code": True, "gpu_memory_utilization": 0.7}
                        })

                        logger.info(f"📉 [Iteration {i}] 부족분 {len(missing_tasks)}건 보완 시작...")
                        current_batch = await self.run_generation_batch(session, missing_tasks, mode="targeted")
                        
                        # 각 Iteration 종료 후 리포트
                        _, q_stats = self.find_imbalanced_tasks(output_file, dataset)
                        p_stats = self.get_final_report()
                        self.print_epoch_report(f"Iteration {i}", q_stats, p_stats)

                        logger.info("🧹 [MCP] VRAM 확보를 위해 모델 언로드 요청...")
                        await session.call_tool("unload_model")

                        self.evaluator.load()
                        try:
                            logger.info(f"⚖️ [Eval_{i}] 로컬 모델 기반 품질 검증 실행")
                            # 동기 함수이므로 to_thread로 실행
                            validated_data = await asyncio.to_thread(
                                self.evaluator.evaluate_batch, 
                                current_batch[:20], 
                                "질문과 답변이 논리적으로 일관되는가?"
                            )
                            avg_score = np.mean([d['eval_score'] for d in validated_data])
                            logger.info(f"⭐ [Initial] LLM Judge 평균 점수: {avg_score:.2f} / 5.0")
                        finally:
                            self.evaluator.unload()

                finally:
                    await self.write_queue.put(None)
                    await save_worker_task

        return output_file

    if __name__ == "__main__":
        config = Config()
        augmentor = AsyncDataAugmentor(config.MCP_URL, config.GEN_HF_MODEL_ID, config)
        asyncio.run(augmentor.run_pipeline_async(dataset, f"{config.AUGMENTED_DATA_PATH}"))

    # def _run_model_generate(self, messages, num_return_sequences) :
    #     model_inputs = self.tokenizer.apply_chat_template(
    #         messages,
    #         tokenize=True,
    #         add_generation_prompt=True,
    #         return_tensors="pt"
    #     ).to(self.accelerator.device)
        
    #     input_ids = model_inputs['input_ids']
    #     attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()
    #     prompt_len = input_ids.shape[-1]

    #     with torch.no_grad() :
    #         outputs = self.model.generate(
    #             input_ids=input_ids,
    #             attention_mask=attention_mask,
    #             max_new_tokens=self.config.MAX_NEW_TOKENS,  
    #             do_sample=True,
    #             temperature=self.config.TEMPERATURE,
    #             top_p=self.config.TOP_P,
    #             top_k=self.config.TOP_K,
    #             repetition_penalty=self.config.REPETITION_PENALTY,
    #             num_return_sequences=self.config.NUM_RETURN_SEQUENCES,
    #             pad_token_id=self.tokenizer.pad_token_id,
    #             eos_token_id=self.tokenizer.eos_token_id
    #         )
        
    #     all_parsed_results = []

    #     for i in range(self.config.NUM_RETURN_SEQUENCES) :
    #         raw_text = self.tokenizer.decode(
    #             outputs[i][prompt_len:],
    #             skip_special_tokens=True # 
    #         )
    #         parsed_result = self._parse_output(raw_text)
    #         all_parsed_results.extend(parsed_result)
        
    #     return all_parsed_results

    # @weave.op()
    # def generate_samples(self, idx, faq_context) :
    #     user_content = LIBRARY_QA_USER_TEMPLATE.replace("{faq_content}", faq_context)
    #     messages = [
    #         {"role": "system", "content": LIBRARY_QA_SYSTEM_PROMPT_COT_FEW_SHOT},
    #         {"role": "user", "content": user_content}
    #     ]
    #     logger.info(messages)
    #     return self._run_model_generate(messages, self.config.NUM_RETURN_SEQUENCES)
    
    # @weave.op()
    # def generate_targeted_samples(self, idx, faq_context, target_labels) :
    #     target_labels_str = ", ".join(target_labels)
    #     all_guidelines = ALL_GUIDELINES
    #     selected_guidelines = "\n".join([all_guidelines[l] for l in target_labels if l in all_guidelines])
    #     selected_examples = ",\n      ".join([ALL_COT_FEW_SHOT_EXAMPLES[l] for l in target_labels if l in ALL_COT_FEW_SHOT_EXAMPLES])

    #     system_content = LIBRARY_QA_TARGETED_SYSTEM_PROMPT_COT_FEW_SHOT.replace("{target_labels}", target_labels_str).replace("{selected_guidelines}", selected_guidelines).replace("{selected_examples}", selected_examples)
    #     user_content = LIBRARY_QA_USER_TEMPLATE.replace("{faq_content}", faq_context)
    #     messages = [
    #         {"role": "system", "content": system_content},
    #         {"role": "user", "content": user_content}
    #     ]
    #     logger.info(messages)
    #     return self._run_model_generate(messages, self.config.NUM_RETURN_TARGET_SEQUENCES)

    # def _run_generation_loop(self, items, output_file, mode="initial") :
    #     if not items :
    #         return
    #     desc = "Initial Gen" if mode == "initial" else "Targeted Gen"
    #     disable_tqdm = not self.accelerator.is_main_process

    #     base, ext = os.path.splitext(output_file)
    #     rank_output_file = f"{base}_rank{self.accelerator.process_index}{ext}"

    #     with open(rank_output_file, "a", encoding="utf-8") as f:
    #         for item in tqdm(items, total=len(items), desc=f"{desc} (Rank {self.accelerator.process_index})", disable=disable_tqdm):
    #             try :
    #                 if mode == "initial" :
    #                     idx, row = item
    #                     faq_context = row['DES']
    #                     result = self.generate_samples(idx, faq_context)
    #                     meta = {'faq_id': idx, 'original_title': row.get('TITLE', '')}
    #                 else: # targeted
    #                     idx = item['idx']
    #                     faq_context = item['context']
    #                     targets = item['targets']
    #                     meta = {'faq_id': idx, 'original_title': item['meta_title']}
    #                     result = self.generate_targeted_samples(idx, faq_context, targets)
                    
    #                 logger.info(result)
    #                 logger.info(meta)
    #                 # 저장
    #                 if result:
    #                     for res in result:
    #                         # 🌟 [핵심] FAQ ID를 데이터에 주입 (나중에 groupby용)
    #                         res.update(meta)
    #                         f.write(json.dumps(res, ensure_ascii=False) + "\n")

    #                         # WandB 로깅 (Main Process만, 첫 번째 샘플만)
    #                         if self.accelerator.is_main_process and len(result) > 0 and result.index(res) == 0:
    #                             self.wandb_table.add_data(
    #                                 res.get('faq_id'), res.get('question'), res.get('answer'), res.get('label')
    #                                 )
    #                     f.flush()
    #                     os.fsync(f.fileno())

    #             except Exception as e:
    #                 import traceback
    #                 logger.error(f"💥 치명적 에러 발생 (idx: {idx if 'idx' in locals() else '?'}): {e}")
    #                 logger.error(traceback.format_exc()) # <-- 이게 범인을 알려줍니다.
                    
    #                 # 디버깅을 위해 여기서 멈추게 하려면 raise를 쓰세요
    #                 raise e

    # def merge_rank_files(self, output_file) :
    #     if not self.accelerator.is_main_process:
    #         return

    #     logger.info("📦 분산된 데이터 파일 병합 중...")
    #     base, ext = os.path.splitext(output_file)

    #     with open(output_file, "a", encoding="utf-8") as outfile:
    #         for rank in range(self.accelerator.num_processes):
    #             rank_file = f"{base}_rank{rank}{ext}"
    #             if os.path.exists(rank_file):
    #                 with open(rank_file, "r", encoding="utf-8") as infile:
    #                     # 파일 내용 복사
    #                     import shutil
    #                     shutil.copyfileobj(infile, outfile)
                    
    #                 try :
    #                     logger.info(f"병합 후 조각 파일 삭제 (중복 병합 방지): {rank_file}")
    #                     os.remove(rank_file) 
    #                 except OSError as e:
    #                     logger.error(f"💥 치명적 에러 발생 (idx: {idx if 'idx' in locals() else '?'}): {e}")
    #                     logger.error(traceback.format_exc()) # <-- 이게 범인을 알려줍니다.
                        
    #     logger.info(f"✅ 병합 완료: {output_file}")
    
    # def run_pipeline(self, dataset, output_file="../data/json/augmented_data.jsonl"):
    #     # --- 1. 초기 생성 (Initial Pass) ---
    #     if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
    #         if self.accelerator.is_main_process:
    #             logger.info("🚀 [Step 1] 초기 데이터 생성 시작...")
    #             with open(output_file, "w", encoding="utf-8") as f: pass
            
    #         # 데이터셋에 인덱스 부여해서 리스트로 변환 (idx, row)
    #         indexed_dataset = list(enumerate(dataset))
            
    #         # 멀티 GPU 분산 (Sharding)
    #         my_items = indexed_dataset[self.accelerator.process_index::self.accelerator.num_processes]
    #         self._run_generation_loop(my_items, output_file, mode="initial")
    #     else:
    #         logger.info("📂 기존 파일이 이미 존재합니다. 데이터를 검수합니다.")

    #     self.accelerator.wait_for_everyone()
    #     self.merge_rank_files(output_file)

    #     # --- 2. 반복 보완 (Iterative Refinement) ---
    #     max_iter = getattr(self.config, 'MAX_AUG_ITERATIONS', 3)
        
    #     for i in range(1, max_iter + 1):
    #         self.accelerator.wait_for_everyone() # 동기화
            
    #         # 정제 및 부족분 분석 (내부적으로 동기화 포함됨)
    #         missing_tasks = self.find_imbalanced_tasks(dataset, output_file)
    #         logger.info(missing_tasks)
            
    #         if not missing_tasks:
    #             if self.accelerator.is_main_process:
    #                 logger.info("✨ 모든 데이터 균형 완료! 증강 종료.")
    #             break
            
    #         if self.accelerator.is_main_process:
    #             logger.info(f"📉 [Step 2-{i}] 부족한 작업: {len(missing_tasks)}건. 추가 생성...")

    #         # 부족분 분산 처리 (Sharding)
    #         my_tasks = missing_tasks[self.accelerator.process_index::self.accelerator.num_processes]
            
    #         if len(my_tasks) > 0:
    #             self._run_generation_loop(my_tasks, output_file, mode="targeted")
            
    #         self.accelerator.wait_for_everyone()
    #         self.merge_rank_files(output_file)

    #     # 완료 로그
    #     if self.accelerator.is_main_process:
    #         if wandb.run:
    #             wandb.log({"generated_qa_samples": self.wandb_table})
    #             logger.info("📊 WandB 업로드 완료")

    #     return output_file