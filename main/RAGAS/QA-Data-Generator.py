import os
import torch
import pandas as pd
import random
import json
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from ragas.testset import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
from ragas.testset.prompts import (
    seed_question_prompt,
    reasoning_question_prompt,
    multi_context_question_prompt,
    question_answer_prompt,
    filter_question_prompt
)

def patch_ragas_prompts_to_korean():
    """Ragas 0.1.21의 실제 기본 영어 프롬프트 지시문을 한국어로 강제 덮어씁니다."""
    
    # 1. simple 질문 생성용 프롬프트
    seed_question_prompt.instruction = (
        "주어진 문맥(context)을 바탕으로 대답할 수 있는 질문을 생성하세요. "
        "모든 출력값은 반드시 자연스러운 **한국어**여야 하며, 영어를 섞지 마세요."
    )
    
    # 2. reasoning 질문 생성용 프롬프트
    reasoning_question_prompt.instruction = (
        "주어진 질문을 바탕으로, 문맥 내에서 논리적인 추론이나 연결이 필요한 더 복잡한 질문으로 재작성하세요. "
        "재작성된 질문은 반드시 **한국어**로만 작성하세요."
    )
    
    # 3. multi_context 질문 생성용 프롬프트
    multi_context_question_prompt.instruction = (
        "주어진 질문을 여러 문맥 조각들을 종합해야만 대답할 수 있는 질문으로 재작성하세요. "
        "재작성된 질문은 반드시 **한국어**로만 작성하세요."
    )
    
    # 4. 정답(ground_truth) 생성용 프롬프트
    question_answer_prompt.instruction = (
        "주어진 문맥 정보를 사용하여 질문에 대한 정답을 작성하세요. "
        "답변은 반드시 **한국어**로 작성하세요."
    )
    
    # 5. 질문 품질 필터링 프롬프트
    filter_question_prompt.instruction = (
        "생성된 질문이 주어진 문맥으로 명확히 답변 가능한지, 그리고 **한국어**로 올바르게 작성되었는지 평가하세요."
    )


def main():
    # 0. 프롬프트 패치 적용
    patch_ragas_prompts_to_korean()

    # 1. vLLM 연결 (32B 모델은 충분히 똑똑하므로 지시만 명확하면 됩니다)
    print("vLLM 연결 중...")
    local_llm = ChatOpenAI(
        model="/models/Exaone-3.5-32B-Instruct",
        openai_api_key="EMPTY",
        openai_api_base="http://localhost:8002/v1",
        max_tokens=2048,
        temperature=0.4,  # 약간 낮춰서 일관성 확보
    )

    # 2. 한국어 임베딩
    print("임베딩 로딩...")
    embeddings = HuggingFaceEmbeddings(
        model_name="dragonkue/snowflake-arctic-embed-l-v2.0-ko",
        model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
    )

    # 3. 데이터 로드 (FAQ 데이터 특화)
    print("데이터 로드...")
    data_path = Path("../data/data.xlsx")
    df = pd.read_excel(data_path)
    
    documents = []
    for i, row in df.iterrows():
        # FAQ 형식을 살려 문맥을 구성
        content = f"제목: {row.get('TITLE', '')}\n내용: {row.get('DES', '')}"
        documents.append(Document(page_content=content, metadata={"faq_id": i}))

    # 4. Generator 초기화
    print("Generator 초기화...")
    generator = TestsetGenerator.from_langchain(
        generator_llm=local_llm,
        critic_llm=local_llm,
        embeddings=embeddings,
    )

    # 5. 질문 분포 설정
    distributions = {
        simple: 0.5,
        reasoning: 0.25,
        multi_context: 0.25,
    }

    print(f"🚀 생성 시작 (test_size=4)...")

    random.shuffle(documents) 
    
    batch_size = 3
    target_size = 200
    generated_count = 0
    output_path = Path("/home/vsc/LLM_TUNE/QA-FineTune/main/data/golden_dataset/ragas_korean.jsonl")

    # 3. 배치 루프 실행
    with open(output_path, 'a', encoding='utf-8') as f:
        # 0부터 110까지 batch_size만큼 건너뛰며 반복
        while generated_count < target_size:
            # 전체 110개 문서 중 무작위로 3개 추출 (Multi-context 유도)
            batch_docs = random.sample(documents, k=min(batch_size, len(documents)))
            
            try:
                # 한 배치당 2~3개의 QA 쌍 생성을 시도합니다.
                curr_testset = generator.generate_with_langchain_docs(
                    batch_docs, 
                    test_size=2, # 배치당 생성 수를 조절하여 품질 유지
                    distributions={simple: 0.5, reasoning: 0.25, multi_context: 0.25}
                )
                
                df_batch = curr_testset.to_pandas()
                for _, row in df_batch.iterrows():
                    # JSONL 저장 (ensure_ascii=False 필수!)
                    f.write(json.dumps(row.to_dict(), ensure_ascii=False) + '\n')
                    generated_count += 1
                
                f.flush()
                print(f"✨ 생성 성공: {generated_count}/{target_size}")

            except Exception as e:
                print(f"⚠️ 배치 생성 실패: {e}")
                continue

    logger.info(f"🏁 생성 완료! 최종 {generated_count}개 데이터가 저장되었습니다.")

if __name__ == "__main__":
    main()