import os
import asyncio
import json
import sys
import aiofiles
import wandb
import weave
import re
import time
import random
from tqdm.asyncio import tqdm
from transformers import AutoTokenizer
from mcp import ClientSession
from mcp.client.sse import sse_client
from contextlib import asynccontextmanager

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from logger_config import get_infer_logger
from prompts import *

logger = get_infer_logger()

class QAPipeline:
    def __init__(self, model_id, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.semaphore = asyncio.Semaphore(getattr(config, 'NUM_SEMAPHORES', 3))
        self.labels = config.LABELS
        
        # 파일 경로 설정
        self.raw_output_file = getattr(config, 'INFER_DATA_PATH', '../data/evaluation/raw_generated.jsonl')
        self.eval_output_file = getattr(config, 'EVALUATION_DATA_PATH', '../data/evaluation/final_evaluated.jsonl')
        
        self.write_queue = asyncio.Queue()
        self.perf_stats = {"total_tokens": 0, "total_time": 0.0, "total_samples": 0}

    def _parse_output(self, raw_output):
        clean_text = re.sub(r'```json\s*|```', '', raw_output).strip()
        json_pattern = re.compile(r'\{.*?\}', re.DOTALL)
        matches = json_pattern.findall(clean_text)
        
        results = []
        for match in matches:
            try:
                results.append(json.loads(match.replace('\n', ' ')))
            except json.JSONDecodeError:
                continue
        return results

    def log_tps_and_sps(self):
        tps = self.perf_stats["total_tokens"] / self.perf_stats["total_time"] if self.perf_stats["total_time"] > 0 else 0
        sec_per_sample = self.perf_stats["total_time"] / self.perf_stats["total_samples"] if self.perf_stats["total_samples"] > 0 else 0
        
        logger.info(
            f"\n{'='*50}\n"
            f"⚡ 생성 성능 리포트\n"
            f"{'-'*50}\n"
            f"🚀 TPS: {tps:.2f} | ⏱️ Sec/Sample: {sec_per_sample:.2f}s\n"
            f"📦 총 생성 샘플: {self.perf_stats['total_samples']}개\n"
            f"{'='*50}"
        )
        if wandb.run:
            wandb.log({"perf/tps": tps, "perf/sec_per_sample": sec_per_sample})

    async def _save_worker(self, filepath):
        logger.info(f"💾 실시간 저장 워커 시작: {filepath}")
        async with aiofiles.open(filepath, mode='a', encoding='utf-8') as f:
            while True:
                result = await self.write_queue.get()
                if result is None: break
                
                items = result if isinstance(result, list) else [result]
                for item in items:
                    await f.write(json.dumps(item, ensure_ascii=False) + "\n")
                
                await f.flush()
                self.write_queue.task_done()

    @asynccontextmanager
    async def get_mcp_session(self, url):
        logger.info(f"⏳ 서버 접속 대기 중... ({url})")
        async with sse_client(url, timeout=300.0, sse_read_timeout=1800.0) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                try:
                    yield session
                finally:
                    pass

    @weave.op()
    async def _mcp_generate_single(self, session, messages, meta):
        async with self.semaphore:
            start_t = time.perf_counter()
            try:
                response = await session.call_tool("generate_text", {
                    "messages": messages,
                    "max_tokens": getattr(self.config, 'MAX_NEW_TOKENS', 512),
                    "temperature": getattr(self.config, 'TEMPERATURE', 0.7)
                })
                raw_text = response.content[0].text
                end_t = time.perf_counter()

                parsed_results = self._parse_output(raw_text)
                
                if parsed_results:
                    for res in parsed_results:
                        res.update(meta)
                        token_count = len(self.tokenizer.encode(res.get('answer', '')))
                        self.perf_stats["total_tokens"] += token_count
                        self.perf_stats["total_samples"] += 1
                    
                    self.perf_stats["total_time"] += (end_t - start_t)
                    await self.write_queue.put(parsed_results)
                
                return parsed_results
            except Exception as e:
                logger.warning(f"⚠️ 추론 중 에러 발생: {e}")
                return []

    async def run_pipeline_async(self, dataset, output_file):
        """해당 기법에 대해 모델 출력 결과 생성 -> 평가 일직선 파이프라인"""
        save_worker_task = asyncio.create_task(self._save_worker(self.raw_output_file))

        try:
            # ==========================================
            # STAGE 1: 모든 상황(DES)에 대해 QA 데이터 생성
            # ==========================================
            logger.info("🚀 [STAGE 1] 데이터 생성 파이프라인 시작")
            
            async with self.get_mcp_session(self.config.GEN_MCP_URL) as session:
                await asyncio.wait_for(
                    session.call_tool("switch_model", {
                        "model_id": self.config.GEN_SERVER_MODEL_NAME,
                        "config": {"trust_remote_code": True, "gpu_memory_utilization": 0.7}
                    }),
                    timeout=600
                )

                async_tasks = []
                for idx, row in enumerate(dataset):
                    faq_context = row.get('DES', '')
                    # 각 상황(Context)마다 지정된 라벨(yes/no/info 등)별로 질문-답변 쌍 생성
                    for target_label in self.labels:
                        meta = {
                            'faq_id': idx, 
                            'original_title': row.get('TITLE', ''), 
                            'target_label': target_label,
                            'context': faq_context
                        }
                        
                        system_content = LIBRARY_QA_SYSTEM_PROMPT_NO_COT.replace("{target_label}", target_label)
                        user_content = LIBRARY_QA_USER_TEMPLATE.replace("{faq_content}", faq_context)
                        messages = [
                            {"role": "system", "content": system_content},
                            {"role": "user", "content": user_content}
                        ]
                        async_tasks.append(self._mcp_generate_single(session, messages, meta))

                logger.info(f"📞 총 {len(async_tasks)}건의 데이터 생성 요청 시작...")
                await tqdm.gather(*async_tasks, desc="🚀 생성 진행 중")

                logger.info("⏳ 파일 저장 대기 중...")
                await self.write_queue.join()
                self.log_tps_and_sps()

                try:
                    logger.info("🧹 생성 모델 언로드...")
                    await asyncio.wait_for(session.call_tool("unload_model", {}), timeout=60)
                except Exception:
                    pass

            # ==========================================
            # STAGE 2: 생성된 데이터를 읽어와서 환각(Hallucination) 정밀 평가
            # ==========================================
            logger.info("🚀 [STAGE 2] 데이터 평가 파이프라인 시작")
            
            # 생성된 데이터 로드
            generated_data = []
            if os.path.exists(self.raw_output_file):
                async with aiofiles.open(self.raw_output_file, mode='r', encoding='utf-8') as f:
                    async for line in f:
                        if line.strip():
                            generated_data.append(json.loads(line))

            if not generated_data:
                logger.error("❌ 평가할 데이터가 없습니다.")
                return None

            async with self.get_mcp_session(self.config.EVAL_MCP_URL) as session:
                await asyncio.wait_for(
                    session.call_tool("switch_model", {
                        "model_id": self.config.EVAL_SERVER_MODEL_NAME,
                        "config": {"trust_remote_code": True, "gpu_memory_utilization": 0.8}
                    }),
                    timeout=600
                )

                random.seed(42)
                sample_size = 20
                random_samples = random.sample(generated_data, sample_size)

                logger.info(f"⚖️ 환각 정밀 품질 검수 시작 (총 {len(random_samples)}건)")
                
                # final_evaluate_batch 툴 호출
                mcp_result = await session.call_tool("final_evaluate_batch", {
                    "data_list": random_samples, 
                    "criteria": "답변에 환각(Hallucination) 현상은 없는지, 답변은 질문에 논리적으로 부합하는지, 또한 실제 사서가 말하듯 정중하게 답변하였는지 엄격하게 평가하라."
                })

                if mcp_result.isError:
                    logger.error(f"❌ 서버 측 에러: {mcp_result.content[0].text}")
                    return None

                # 결과 파싱
                validated_data = []
                for content_item in mcp_result.content:
                    if getattr(content_item, 'type', None) == 'text':
                        try:
                            item_json = json.loads(content_item.text)
                            if isinstance(item_json, list): validated_data.extend(item_json)
                            else: validated_data.append(item_json)
                        except json.JSONDecodeError:
                            continue

                if validated_data:
                    # 최종 평가 결과 파일 저장
                    async with aiofiles.open(self.eval_output_file, mode='w', encoding='utf-8') as f:
                        for entry in validated_data:
                            await f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                    logger.info(f"💾 [완료] 평가 결과 {len(validated_data)}건 저장 완료: {self.eval_output_file}")

                    # W&B 및 로그 출력용 점수 집계
                    avg_score = sum(d.get('eval_score', 0) for d in validated_data) / len(validated_data)
                    avg_faith = sum(d.get('eval_faithfulness', 0) for d in validated_data) / len(validated_data)
                    avg_rel = sum(d.get('eval_relevancy', 0) for d in validated_data) / len(validated_data)
                    avg_pers = sum(d.get('eval_persona', 0) for d in validated_data) / len(validated_data)

                    logger.info(f"⭐ [최종 리포트] 평균 총점: {avg_score:.2f}/15.0")
                    logger.info(f"   ㄴ 환각방지(8): {avg_faith:.2f} | 논리성(5): {avg_rel:.2f} | 말투(2): {avg_pers:.2f}")

                    if wandb.run:
                        wandb.log({
                            "eval/avg_total_score": avg_score,
                            "eval/avg_faithfulness": avg_faith,
                            "eval/avg_relevancy": avg_rel,
                            "eval/avg_persona": avg_pers,
                            "eval/total_samples": len(validated_data)
                        })

                try:
                    logger.info("🧹 평가 모델 언로드...")
                    await asyncio.wait_for(session.call_tool("unload_model", {}), timeout=60)
                except Exception:
                    pass

            return self.eval_output_file # 최종 평가된 파일 경로 반환

        except Exception as pipe_err: 
            logger.error(f"❌ 파이프라인 실행 중 오류 발생: {pipe_err}", exc_info=True)

        finally: 
            await self.write_queue.put(None)
            await save_worker_task
            logger.info("✅ 파이프라인 프로세스 완전 종료")


if __name__ == "__main__":
    # 실행부 예시 (main.py에서 임포트해서 사용하는 것을 권장)
    pass