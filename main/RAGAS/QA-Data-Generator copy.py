import os
import sys
import torch
import pandas as pd
import random
import json
import ragas.testset.prompts as ragas_prompts
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from ragas.testset import TestsetGenerator # src/ragas/testset
from ragas.testset.evolutions import simple, reasoning, multi_context #src/ragas/testset/evolutions
from prompts_ko import seed_question_prompt_ko, reasoning_question_prompt_ko, multi_context_question_prompt_ko, question_answer_prompt_ko
from ragas.testset.extractor import KeyphraseExtractor
from ragas.run_config import RunConfig

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from logger_config import get_data_logger
logger = get_data_logger()

def apply_korean_template():
    """Ragas의 4대 메인 프롬프트를 한국어 버전(prompts_ko)으로 덮어씁니다."""
    # 기초 질문 패치
    ragas_prompts.seed_question_prompt.instruction = seed_question_prompt_ko.instruction
    ragas_prompts.seed_question_prompt.examples = seed_question_prompt_ko.examples
    ragas_prompts.seed_question_prompt.language = "korean"

    # 추론형 패치
    ragas_prompts.reasoning_question_prompt.instruction = reasoning_question_prompt_ko.instruction
    ragas_prompts.reasoning_question_prompt.examples = reasoning_question_prompt_ko.examples
    ragas_prompts.reasoning_question_prompt.language = "korean" 
    
    # 다중 문맥 패치
    ragas_prompts.multi_context_question_prompt.instruction = multi_context_question_prompt_ko.instruction
    ragas_prompts.multi_context_question_prompt.examples = multi_context_question_prompt_ko.examples
    ragas_prompts.multi_context_question_prompt.language = "korean" 
    
    # 정답 생성 패치
    ragas_prompts.question_answer_prompt.instruction = question_answer_prompt_ko.instruction
    ragas_prompts.question_answer_prompt.examples = question_answer_prompt_ko.examples
    ragas_prompts.question_answer_prompt.language = "korean"

def patch_extractor_to_ko(generator):
    """
    KeyphraseExtractor의 프롬프트 구조를 Ragas의 기대치(JSON Schema)에 맞춰 한국어로 교체합니다.
    """
    extractor = generator.docstore.extractor
    original_extract = extractor.extract
    
    if isinstance(extractor, KeyphraseExtractor):
        p = extractor.extractor_prompt
        
        p.language = "korean"
        p.instruction = (
            "주어진 문맥에서 가장 중요한 핵심 키워드(keyphrases)를 3~5개 추출하세요. "
            "결과는 반드시 'keyphrases'라는 키를 가진 JSON 형식으로 한국어로만 출력하세요."
        )
        
        # 🚀 핵심: 'output' 안에 'keyphrases' 객체를 명시적으로 포함!
        p.examples = [
            {
                "text": "아람누리도서관은 고양시의 대표적인 예술 특화 도서관입니다.",
                "output": {
                    "keyphrases": ["아람누리도서관", "고양시", "예술 특화"]
                }
            },
            {
                "text": "메이커스페이스는 3D 프린터와 레이저 커터 등 다양한 장비를 이용할 수 있는 창작 공간입니다.",
                "output": {
                    "keyphrases": ["메이커스페이스", "3D 프린터", "창작 공간"]
                }
            }
        ]
        logger.info("✅ Extractor 구조 교정 및 한국어 패치 완료!")

    async def debug_extract(node, is_async=True):
        # 실제 추출 수행
        kp = await original_extract(node, is_async)
        if kp :
            logger.info(f"🤗 키워드 추출 성공! 추출된 키워드 : {kp}")
        if not kp:
            logger.warning(f"⚠️ 키워드 추출 실패! 지문 내용: {node.page_content[:30]}...")
        return kp

    extractor.extract = debug_extract

def eradicate_english_prompts(generator):
    """
    Import 없이 generator 내부 메모리를 재귀적으로 뒤져서 
    숨겨진 모든 Prompt 객체를 찾아내 한국어로 강제 변환합니다.
    """
    visited = set()

    def traverse_and_patch(obj):
        # 무한 루프 방지
        if id(obj) in visited:
            return
        visited.add(id(obj))

        # 🎯 목표물 발견: 클래스 이름이 'Prompt'이고 instruction이 있는 경우
        if obj.__class__.__name__ == 'Prompt' and hasattr(obj, 'instruction'):
            # 이미 우리가 한글화한 4대 메인 프롬프트와 Extractor는 건너뜁니다
            safe_names = [
                "seed_question", "answer_formulate", "reasoning_question", 
                "multi_context_question", "keyphrase_extraction"
            ]
            
            if hasattr(obj, 'name') and obj.name not in safe_names:
                obj.language = "korean"
                obj.instruction += "\n\n[CRITICAL RULE] You MUST process and output everything entirely in KOREAN. Do NOT use English."
                
                if hasattr(obj, 'output_format_instruction') and obj.output_format_instruction:
                    obj.output_format_instruction += "\n모든 JSON의 value 값은 반드시 한국어로 작성하세요."
                
                # 영어 예시가 모델을 오염시키지 않도록 강제 삭제
                if hasattr(obj, 'examples'):
                    obj.examples = []
                    
                logger.info(f"🔫 숨은 프롬프트 한국어 강제 패치 완료: {obj.name}")

        # 🔍 내부 속성(Attributes) 및 자료구조 재귀 탐색
        if hasattr(obj, '__dict__'):
            for val in obj.__dict__.values():
                traverse_and_patch(val)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                traverse_and_patch(item)
        elif isinstance(obj, dict):
            for val in obj.values():
                traverse_and_patch(val)

    # 사냥 시작
    traverse_and_patch(generator)
    logger.info("🛡️ Ragas 내부의 모든 영어 프롬프트 제거")

def verify_prompts(generator):
    """현재 generator가 사용 중인 프롬프트들이 한국어인지 출력하여 확인합니다."""
    logger.info("\n🔍 [프롬프트 최종 검증 시작]")
    
    # 1. Seed Question 확인
    logger.info(f"1. Seed Question 지시문: {ragas_prompts.seed_question_prompt.instruction[:30]}...")
    logger.info(f"   언어 설정: {ragas_prompts.seed_question_prompt.language}")
    
    # 2. Extractor 확인
    extractor_p = generator.docstore.extractor.extractor_prompt
    logger.info(f"2. Extractor 지시문: {extractor_p.instruction[:30]}...")
    logger.info(f"   예시(첫번째): {extractor_p.examples[0]['output']}")
    
    # 3. Question Answer 확인 (정답 생성용)
    logger.info(f"3. 정답 생성 지시문: {ragas_prompts.question_answer_prompt.instruction[:30]}...")

def main():
    apply_korean_template()
    # 1. vLLM 연결 (32B 모델은 충분히 똑똑하므로 지시만 명확하면 됩니다)
    logger.info("🔗 vLLM 연결 중...")
    local_llm = ChatOpenAI(
        model="/models/Exaone-3.5-32B-Instruct",
        openai_api_key="EMPTY",
        openai_api_base="http://localhost:8002/v1",
        max_tokens=2048,
        temperature=0.4, 
    )

    # 2. 한국어 임베딩
    logger.info("💾 임베딩 로딩...")
    embeddings = HuggingFaceEmbeddings(
        model_name="dragonkue/snowflake-arctic-embed-l-v2.0-ko",
        model_kwargs={'device': 'cpu'},
    )

    # 3. 데이터 로드 (FAQ 데이터 특화)
    logger.info("📂 데이터 로드...")
    data_path = Path("../data/data.xlsx")
    df = pd.read_excel(data_path)
    
    documents = []
    for i, row in df.iterrows():
        metadata = {
            "faq_id": i,
            "filename": f"faq_row_{i}",  # 각 행에 고유 이름을 부여
            "source": "manual_excel"
        }
        content = f"제목: {row.get('TITLE', '')}\n내용: {row.get('DES', '')}"
        documents.append(Document(page_content=content, metadata=metadata))

    run_config = RunConfig(
        max_workers=8,
        seed=42
    )

    # 4. Generator 초기화
    logger.info("🏗️ Generator 초기화...")
    generator = TestsetGenerator.from_langchain(
        generator_llm=local_llm,
        critic_llm=local_llm,
        embeddings=embeddings,
        run_config=run_config
    )
    patch_extractor_to_ko(generator)
    eradicate_english_prompts(generator)
    verify_prompts(generator)

    # 5. 질문 분포 설정
    distributions = {
        simple: 0.5,
        reasoning: 0.25,
        multi_context: 0.25,
    }

    random.shuffle(documents) 
    
    batch_size = 20
    test_batch_size = 2
    target_size = 200
    generated_count = 0
    output_path = Path("/home/vsc/LLM_TUNE/QA-FineTune/main/data/golden_dataset/ragas_korean.jsonl")

    logger.info(f"🚀 생성 시작 (test_size={target_size})...")

    # 3. 배치 루프 실행
    with open(output_path, 'a', encoding='utf-8') as f:
        # 0부터 110까지 batch_size만큼 건너뛰며 반복
        while generated_count < target_size:
            # 전체 110개 문서 중 무작위로 20개 추출 (Multi-context 유도)
            batch_docs = random.sample(documents, k=min(batch_size, len(documents)))
            
            try:
                # 한 배치당 4개의 QA 쌍 생성을 시도
                curr_testset = generator.generate_with_langchain_docs(
                    batch_docs, 
                    test_size=test_batch_size,
                    distributions=distributions
                )
                
                df_batch = curr_testset.to_pandas()
                for _, row in df_batch.iterrows():
                    # JSONL 저장 (ensure_ascii=False 필수!)
                    f.write(json.dumps(row.to_dict(), ensure_ascii=False) + '\n')
                    generated_count += 1
                
                f.flush()
                logger.info(f"✨ 생성 성공: {generated_count}/{target_size}")

            except Exception as e:
                logger.info(f"⚠️ 배치 생성 실패: {e}")
                continue

    logger.info(f"🏁 생성 완료! 최종 {generated_count}개 데이터가 저장되었습니다.")

if __name__ == "__main__":
    main()