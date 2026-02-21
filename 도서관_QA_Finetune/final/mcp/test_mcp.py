import asyncio
from mcp import ClientSession
from mcp.client.sse import sse_client

# 1. DGX 서버 주소 (포트와 /sse 경로를 꼭 확인하세요!)
# 도커 실행 시 -p 8000:4200으로 하셨다면 아래 주소가 맞습니다.
DGX_SERVER_URL = "http://localhost:8000/sse"

async def main():
    print(f"🔗 {DGX_SERVER_URL} 접속 시도 중...")
    
    async with sse_client(DGX_SERVER_URL) as (read, write):
        async with ClientSession(read, write) as session:
            # MCP 세션 초기화
            await session.initialize()
            print("✅ DGX MCP 서버 연결 완료!")

            # --- [실험 1: 모델 로드 - EXAONE] ---
            print("\n🚀 [STAGE 1] 모델 로드 요청: ")
            # switch_model 도구 호출
            load_result = await session.call_tool("switch_model", {
                "model_name": "A.X-4.0-Light",
                "config": {"gpu_memory_utilization": 0.6} # 메모리 점유율 조절
            })
            print(f"📡 서버 응답: {load_result}")

            # --- [실험 2: 텍스트 생성] ---
            print("\n💬 [STAGE 2] 추론 테스트")
            gen_result = await session.call_tool("generate_text", {
                "prompt": "DGX Spark 서버의 장점을 한 문장으로 말해줘."
            })
            print(f"🤖 AI 답변: {gen_result}")

            # --- [실험 3: 모델 교체 (기존 모델 자동 Unload)] ---
            # 사용자님의 코드 내부에서 switch_model 호출 시 unload_model()이 자동 실행됩니다.
            print("\n🔄 [STAGE 3] 모델 교체 요청 (Exaone-3.5-7.8B로 스위칭)")
            switch_result = await session.call_tool("switch_model", {
                "model_name": "Exaone-3.5-7.8B-Instruct"
            })
            print(f"📡 서버 응답: {switch_result}")

            print("\n💬 [STAGE 4] 추론 테스트")
            gen_result = await session.call_tool("generate_text", {
                "prompt": "안녕, 너가 누구인지 알려줄 수 있을까나?"
            })
            print(f"🤖 AI 답변: {gen_result}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"❌ 에러 발생: {e}")