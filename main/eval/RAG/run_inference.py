import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# 모델 설정 (Docker 환경 변수에서 로드)
MODEL_ID = os.getenv("MODEL_ID", "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct")

print(f"🚀 모델 로딩 시작: {MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
print("✅ 모델 로딩 완료!")

# 1. MCP 검색 도구 호출 함수
async def get_context_from_mcp(question: str):
    # 검색 서버(search-data) 컨테이너와 stdio 방식으로 통신한다고 가정
    server_params = StdioServerParameters(
        command="python",
        args=["search_contents.py"], # 검색 서버 실행 파일
        env=os.environ.copy()
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            # 검색 서버에 등록된 'search_library' 도구 호출
            result = await session.call_tool("search_library", {"query": question})
            return result.content[0].text

# 2. 답변 생성 로직
def generate_answer(question: str, context: str):
    # EXAONE 3.5 전용 프롬프트 템플릿
    prompt = f"""[System]
        당신은 질문에 답변하는 도움이 되는 AI 어시스턴트입니다. 
        주어진 문맥(Context)을 바탕으로 질문에 답하세요. 답을 모른다면 모른다고 답변하세요.

        [Context]
        {context}

        [Question]
        {question}

        [Answer]
    """

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs, 
        max_new_tokens=512, 
        temperature=0.7, 
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id
    )
    
    return tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)

# 3. 메인 실행 루프
async def main():
    question = "상호 대차 서비스는 1인당 몇 권까지 신청 가능한가요?"
    print(f"❓ 질문: {question}")
    
    # 문맥 가져오기 (MCP 검색 서버 활용)
    context = await get_context_from_mcp(question)
    print(f"📚 검색된 문맥: {context[:100]}...")
    
    # 답변 생성
    answer = generate_answer(question, context)
    print(f"🤖 답변: {answer}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())