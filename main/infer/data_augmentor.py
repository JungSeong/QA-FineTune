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
import random
from datetime import datetime
from tqdm.asyncio import tqdm
from transformers import AutoTokenizer
from mcp import ClientSession
from mcp.client.sse import sse_client
from kiwipiepy import Kiwi
from prompts import *
from scipy.stats import entropy
from contextlib import asynccontextmanager
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from httpx import Timeout

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from logger_config import get_infer_logger
logger = get_infer_logger()

class AsyncDataAugmentor:
    def __init__(self, model_id, config):
        self.model_id = model_id
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.semaphore = asyncio.Semaphore(getattr(config, 'NUM_SEMAPHORES', 3))
        self.labels = config.LABELS
        self.write_queue = asyncio.Queue()
        self.output_file = getattr(config, 'AUGMENTED_DATA_PATH')
        self.kiwi = Kiwi() # 형태소 분석기 초기화
        self.perf_stats = {
            "total_tokens": 0,
            "total_time": 0.0,
            "total_samples": 0
        }
        self.gen_mcp_url = config.GEN_MCP_URL
        self.eval_mcp_url = config.EVAL_MCP_URL
        self.sbert_model = SentenceTransformer('jhgan/ko-sroberta-multitask') # 한국어 임베딩 모댈

    def _parse_output(self, raw_output):
        """json의 출력을 정제해주는 함수"""
        clean_text = re.sub(r'```json\s*|```', '', raw_output).strip()
        
        # 2. 정규표현식으로 { ... } 패턴을 모두 추출 (가장 확실한 방법)
        json_pattern = re.compile(r'\{.*?\}', re.DOTALL)
        matches = json_pattern.findall(clean_text)
        
        results = []
        for match in matches:
            try:
                # 3. 개행 문자 등으로 깨진 JSON 수정 후 로드
                item = json.loads(match.replace('\n', ' '))
                results.append(item)
            except json.JSONDecodeError:
                # 파싱 실패 시 로그를 남기거나 건너뜀
                continue
                
        return results

    def calculate_kl_divergence(self, counts_list):
        """
        현재까지 생성된 데이터에 대해 KL-Divergence값을 계산해주는 함수 
        """
        # 1. 현재 분포 (P) 계산
        counts = np.array(counts_list)
        p = counts / counts.sum()
        
        # 2. 이상적인 균등 분포 (Q) 생성 (모든 라벨의 확률이 동일함)
        n_labels = len(counts)
        q = np.ones(n_labels) / n_labels
        
        # 3. KL Divergence 계산 (P || Q)
        kl_div = entropy(p, q)
        
        return float(entropy(p, q))

    def calculate_semantic_similarity(self, df, column='question', sample_size=100):
        """
        SBERT를 이용해 생성된 데이터의 의미적 중복도(평균 코사인 유사도)를 계산합니다.
        값이 낮을수록 데이터가 의미적으로 다양함을 뜻합니다.
        """
        if len(df) < 2: 
            return 0.0
            
        # 최대 sample_size 만큼 무작위 추출하여 계산 속도 확보
        texts = df[column].dropna().sample(min(len(df), sample_size)).tolist()
        
        # 1. 문장들을 임베딩 벡터로 변환
        embeddings = self.sbert_model.encode(texts, show_progress_bar=True)
        
        # 2. 벡터 간 코사인 유사도 행렬 계산 (NxN 행렬)
        sim_matrix = cosine_similarity(embeddings)
        
        # 3. 자기 자신과의 비교(대각선 1.0) 및 중복 비교를 제외한 '상삼각행렬'의 값만 추출
        upper_tri_indices = np.triu_indices_from(sim_matrix, k=1)
        pairwise_similarities = sim_matrix[upper_tri_indices]
        
        # 4. 평균 유사도 반환
        mean_sim = np.mean(pairwise_similarities)
        return float(mean_sim)

    def calculate_metrics(self, df):
        """
        KL-Divergence & semantic_similarity 값을 한 번에 계산해주는 함수
        """
        if len(df) == 0:
            return 0.0, 0.0
        # 1. 라벨 빈도수를 내부에서 직접 추출 (reindex로 모든 라벨 포함)
        counts = df['label'].value_counts().reindex(self.labels, fill_value=0).tolist()
        # 2. KL-Divergence 계산
        kl_divergence_val = self.calculate_kl_divergence(counts)
        # 3. Semantic Similarity 계산
        semantic_similarity = self.calculate_semantic_similarity(df)
        
        return kl_divergence_val, semantic_similarity

    def find_imbalanced_tasks(self, file_path, original_dataset):
        """
        현재까지 생성된 데이터의 KL-Divergence, semantic_similarity, 생성된 데이터 개수를 반환하고 로깅해주며, 부족한 라벨을 찾아 missing_tasks로 반환해주는 함수
        """
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

        kl_divergence_val, semantic_similarity = self.calculate_metrics(df)

        # 3. 📝 로그 출력 (더 상세하게)
        logger.info(f"\n{'═'*50}\n"
                    f"📈 데이터셋 리포트 (총 {total_count}개)\n"
                    f"{'-'*50}\n"
                    f"{label_stats.to_string()}\n"
                    f"{'-'*50}\n"
                    f"🧩 문장 간 다양성 지수 (Semantic Similarity): {semantic_similarity:.3f} (0에 가까울수록 다양한 문장 생성)\n"
                    f"⚖️ 라벨 간 비율 측정 (KL-Divergence): {kl_divergence_val:.3f} (0에 가까울수록 라벨이 균등하게 생성)\n"
                    f"{'═'*50}")

        # 4. WandB 로깅
        if wandb.run:
            wandb.log({
                "quality/KL-Divergence": kl_divergence_val,
                "quality/semantic_similarity": semantic_similarity,
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
        return missing_tasks, {"kl-divergence": kl_divergence_val, "semantic_similarity": semantic_similarity, "count": total_count}

    def log_tps_and_sps(self):
        """
        TPS (Tokens-Per-Second) & SPS (Samples-Per-Second)를 측정하고 로깅해주는 함수
        """
        tps = self.perf_stats["total_tokens"] / self.perf_stats["total_time"] if self.perf_stats["total_time"] > 0 else 0
        sec_per_sample = self.perf_stats["total_time"] / self.perf_stats["total_samples"] if self.perf_stats["total_samples"] > 0 else 0
        walltime = self.perf_stats["total_time"]
        
        report = (
            f"\n{'='*50}\n"
            f"⚡ 실시간 생성 성능 리포트\n"
            f"{'-'*50}\n"
            f"🚀 TPS (Tokens/Sec): {tps:.2f}\n"
            f"⏱️ Sec/Sample: {sec_per_sample:.2f}s\n"
            f"⏱️ Wall Time: {walltime:.2f}s\n"
            f"🔢 총 생성 토큰: {self.perf_stats['total_tokens']}\n"
            f"📦 총 생성 샘플: {self.perf_stats['total_samples']}\n"
            f"{'='*50}"
        )
        logger.info(report)
        return {"tps": tps, "sec_per_sample": sec_per_sample}

    def log_epoch_report(self, iteration_name, quality_stats, perf_stats):
        """
        매 EPochs 종료 시 출력될 통합 리포트
        """
        logger.info(f"\n{'#'*60}\n"
                    f"📢 [{iteration_name}] 단계 완료 리포트\n"
                    f"{'-'*60}\n"
                    f"📊 [품질] KL-Divergence: {quality_stats['kl-divergence']:.3f} | semantic_similarity: {quality_stats['semantic_similarity']:.3f}\n"
                    f"⚡ [성능] TPS: {perf_stats['tps']:.2f} | Sec/Sample: {perf_stats['sec_per_sample']:.2f}s\n"
                    f"📈 [누적] 총 샘플 수: {quality_stats['count']}개\n"
                    f"{'#'*60}")
        
        # WandB에 단계별 기록
        if wandb.run:
            wandb.log({
                "epoch": iteration_name,
                "metrics/KL-Divergence": quality_stats['kl-divergence'],
                "metrics/semantic_similarity": quality_stats['semantic_similarity'],
                "perf/tps": perf_stats['tps'],
                "perf/sec_per_sample": perf_stats['sec_per_sample']
            })

    async def _save_worker(self):
        """
        현재까지 생성된 데이터의 KL-Divergence, semantic_similarity, 생성된 데이터 개수를 반환하고 로깅해주는 함수
        """
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
        """
        MCP 서버를 통해 추론하고, meta(context 포함)를 결과물에 병합하여 큐에 넣음
        """
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
                        # 🌟 중요: meta에 들어있는 context, faq_id 등을 모든 결과에 주입
                        res.update(meta)
                        
                        # 성능 측정
                        token_count = len(self.tokenizer.encode(res.get('answer', '')))
                        self.perf_stats["total_tokens"] += token_count
                        self.perf_stats["total_samples"] += 1
                    
                    self.perf_stats["total_time"] += (end_t - start_t)
                    # 파일 저장 워커로 데이터 전송
                    await self.write_queue.put(parsed_results)
                
                return parsed_results
            except Exception as e:
                logger.warning(f"⚠️ 추론 중 에러 발생: {e}")
                return []

    async def run_generation_batch(self, session, tasks, mode="initial"):
        """
        데이터셋의 각 항목에 대해 컨텍스트를 추출하고 병렬 생성을 요청
        """
        async_tasks = []
        for item in tasks:
            if mode == "initial":
                idx, row = item[0], item[1]
                faq_context = row['DES']
                # 🌟 원본 컨텍스트를 meta에 담아 나중에 Judge가 볼 수 있게 함
                for target_label in self.labels :
                    meta = {
                        'faq_id': idx, 
                        'original_title': row.get('TITLE', ''), 
                        'target_label': target_label,
                        'context': faq_context, 
                        'iteration': mode
                    }
                    system_content = LIBRARY_QA_SYSTEM_PROMPT_NO_COT.replace("{target_label}", target_label)
                    user_content = LIBRARY_QA_USER_TEMPLATE.replace("{faq_content}", faq_context)
                    messages = [
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": user_content}
                    ]
                    logger.info(f"messages || {messages}")
                    async_tasks.append(self._mcp_generate_single(session, messages, meta))
            else:
                # Targeted(부족분 채우기) 모드 로직
                idx = item['idx']
                faq_context = item['context']
                target_labels = item['targets']

                for target_label in target_labels :
                    meta = {
                        'faq_id': idx, 
                        'original_title': item.get('meta_title', ''),
                        'target_labels': target_label,
                        'context': faq_context, 
                        'iteration': mode,
                    }

                    system_content = LIBRARY_QA_TARGETED_SYSTEM_PROMPT_NO_COT.replace("{target_label}", target_label)
                    user_content = LIBRARY_QA_USER_TEMPLATE.replace("{faq_content}", faq_context)
                    messages = [
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": user_content}
                    ]
                
                    async_tasks.append(self._mcp_generate_single(session, messages, meta))

        # 🚀 tqdm으로 진행 상황을 보며 병렬 실행
        results = await tqdm.gather(*async_tasks, desc=f"🚀 {mode} 증강 진행 중 - 현재 MAX_NEW_TOKENS : {getattr(self.config, 'MAX_NEW_TOKENS', 512)}")
        return [res for sublist in results for res in sublist]

    @asynccontextmanager
    async def get_mcp_session(self, url):
        logger.info(f"⏳ {self.config.EVAL_SERVER_MODEL_NAME} 로딩을 시작합니다. (최대 5분 소요될 수 있습니다...)")

        # float 초 단위로만 → 내부에서 read=sse_read_timeout과 잘 조합됨
        initial_timeout_seconds = 300.0   # SSE 초기 연결을 5분 까지 기다리기

        async with sse_client(
            url,
            timeout=initial_timeout_seconds,       # ← float으로만!
            sse_read_timeout=1800.0,               # SSE 연결 이후 30분까지 대기
        ) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                try:
                    yield session
                finally:
                    pass

    async def run_pipeline_async(self, dataset, output_file):
        """최종 파이프라인"""
        save_worker_task = asyncio.create_task(self._save_worker())

        try: # [1. 전체 파이프라인 시작]
            for i in range(2):
                iter_name = f"Iteration_{i+1}"
                logger.info(f"🚀 [{iter_name}] 파이프라인 시작")

                # 1. 상태 분석 (부족한 태스크 찾기)
                missing_tasks, quality_stats = self.find_imbalanced_tasks(output_file, dataset)
                
                # 생성 모드 결정
                if i == 0 and quality_stats['count'] == 0:
                    mode = "initial"
                    current_tasks = list(enumerate(dataset)) # 전체 생성
                else:
                    mode = "targeted"
                    current_tasks = missing_tasks # 부족한 것만 생성

                # --- [STAGE 1] 데이터 생성 (8000번 포트) ---
                if current_tasks:
                    logger.info(f"📞 생성 서버 접속: {self.config.GEN_MCP_URL}")
                    async with self.get_mcp_session(self.config.GEN_MCP_URL) as session:
                        await asyncio.wait_for(
                            session.call_tool("switch_model", {
                                "model_id": self.config.GEN_SERVER_MODEL_NAME,
                                "config": {"trust_remote_code": True, "gpu_memory_utilization": 0.7}
                            }),
                            timeout=600
                        )

                        logger.info(f"📞 [{mode.upper()}] 모드로 생성 시작 (대상: {len(current_tasks)}건)")
                        await self.run_generation_batch(session, current_tasks, mode=mode)

                        logger.info("⏳ 모든 데이터가 저장될 때까지 대기 중 (Queue Join)...")
                        await self.write_queue.join()

                        perf_stats = self.log_tps_and_sps()
                        
                        try:
                            logger.info("🧹 생성 모델 VRAM 언로드...")
                            await session.call_tool("unload_model")
                        except (anyio.ClosedResourceError, Exception):
                            logger.warning("⚠️ 생성 서버 세션이 이미 종료되어 언로드를 건너뜁니다.")
                else:
                    logger.info("✅ 모든 라벨이 충분합니다. 생성을 건너뜁니다.")
                    perf_stats = {"tps": 0.0, "sec_per_sample": 0.0}
                
                # --- [DATA LOAD] 생성된 혹은 기존 데이터를 메모리로 로드 ---
                new_data = [] # 평가에 사용할 데이터를 담을 리스트
                if os.path.exists(output_file):
                    async with aiofiles.open(output_file, mode='r', encoding='utf-8') as f:
                        async for line in f:
                            if line.strip():
                                new_data.append(json.loads(line))
                
                if not new_data:
                    logger.error("❌ 처리할 데이터가 없습니다.")
                    return

                if new_data:
                    # --- [STAGE 2] 품질 검수 (8001번 포트) ---
                    logger.info(f"📞 평가 서버 접속: {self.config.EVAL_MCP_URL}")

                    async with self.get_mcp_session(self.config.EVAL_MCP_URL) as session: 
                        await session.call_tool("switch_model", {
                            "model_id": self.config.EVAL_SERVER_MODEL_NAME,
                            "config": {"trust_remote_code": True, "gpu_memory_utilization": 0.8}
                        })
                        
                        try: # [2. 품질 검수 내부 try - session 안쪽에 위치]
                            logger.info("⚖️ 데이터 품질 검수 시작")
                            sample_size = 20
                            random_samples = random.sample(new_data, sample_size)

                            mcp_result = await session.call_tool("evaluate_batch", {
                                "data_list": random_samples, 
                                "criteria": "질문과 답변이 논리적으로 일관되었으며, 질문과 라벨이 서로 올바르게 논리적으로 연관되어 있는가?"
                            })

                            if mcp_result.isError:
                                logger.error(f"❌ 서버 측 상세 에러: {mcp_result.content[0].text}")
                                return

                            validated_data = []
                            for content_item in mcp_result.content:
                                if getattr(content_item, 'type', None) == 'text':
                                    try:
                                        item_json = json.loads(content_item.text)
                                        validated_data.append(item_json)
                                    except json.JSONDecodeError as e:
                                        logger.warning(f"⚠️ 개별 항목 JSON 파싱 실패: {e}")
                                        continue

                            # 🌟 중요: 아래 처리 로직들도 session 블록과 try 블록 내부에 정렬되어야 함
                            logger.info(f"📊 서버 응답 수집 완료: 총 {len(validated_data)}건의 데이터를 확인했습니다.")

                            if validated_data:
                                sample = validated_data[0]
                                logger.info(f"🔍 [Sample Evaluation Result Detail]")
                                logger.info(f"📝 질문: {sample.get('question', '')[:50]}...")
                                logger.info(f"🌍 상황: {sample.get('context', '')[:50]}...")
                                logger.info(f"👉 답변: {sample.get('answer', '')[:50]}...")
                                logger.info(f"💡 점수: {sample.get('eval_score', 0)} / 15")
                                logger.info(f"📑 근거: {sample.get('eval_reason', '정보 없음')}")
                                logger.info(f"🤔 추론 과정: {sample.get('eval_thought', '')[:150]}...")

                                eval_output_path = "/home/vsc/LLM_TUNE/QA-FineTune/main/data/json/evaluation_result_big.jsonl"
                                async with aiofiles.open(eval_output_path, mode='a', encoding='utf-8') as f:
                                    for entry in validated_data:
                                        await f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                                logger.info(f"💾 [완료] 원본 및 평가 결과 {len(validated_data)}건 저장 완료: {eval_output_path}")

                                scores = [d.get('eval_score', 0) for d in validated_data if 'eval_score' in d]
                                if scores:
                                    avg_score = sum(scores) / len(scores)
                                    logger.info(f"⭐ [Summary] LLM Judge 평균 점수: {avg_score:.2f} / 15.0")
                                    
                                    if wandb.run:
                                        wandb.log({"eval/avg_score": avg_score, "eval/total_count": len(validated_data)})
                        except Exception as e:
                            logger.error(f"❌ 품질 검수 중 에러: {e}")
                        finally :
                            try:
                                logger.info("🧹 평가 모델 VRAM 언로드...")
                                await session.call_tool("unload_model")
                            except (anyio.ClosedResourceError, Exception):
                                logger.warning("⚠️ 평가 서버 세션이 이미 종료되었습니다.")

                    _, final_quality_stats = self.find_imbalanced_tasks(output_file, dataset)
                    self.log_epoch_report(iter_name, final_quality_stats, perf_stats)

            return output_file # 최종 파일 경로 반환

        # [3. 전체 파이프라인 에러 catch - Top-level try에 대응]
        except Exception as pipe_err: 
            logger.error(f"❌ 파이프라인 실행 중 오류 발생: {pipe_err}", exc_info=True)

        # [4. 어떤 경우에도 반드시 실행 - 워커 종료]
        finally: 
            await self.write_queue.put(None)
            await save_worker_task
            logger.info("✅ 파이프라인 프로세스 종료")

    if __name__ == "__main__":
        config = Config()
        augmentor = AsyncDataAugmentor(config.GEN_HF_MODEL_ID, config)
        # asyncio.run(augmentor.run_pipeline_async(dataset, f"{config.AUGMENTED_COT_DATA_PATH}"))
        asyncio.run(augmentor.run_pipeline_async(dataset, f"{config.AUGMENTED_COT_DATA_PATH}"))