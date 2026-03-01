import asyncio
from mcp import ClientSession
from mcp.client.sse import sse_client

async def test_retrieval():
    # search-data 서비스의 외부 포트(예: 8080)를 사용하세요.
    url = "http://localhost:8080/sse" 
    
    async with sse_client(url) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # 우리가 만든 'search_library' 도구 호출
            question = "상호대차 서비스 이용 방법 알려줘"
            result = await session.call_tool("search_library", {"query": question})
            print(result)
            
            print(f"🔍 질문: {question}")
            print(f"📚 검색된 문맥:\n{result.content[0].text}")

if __name__ == "__main__":
    asyncio.run(test_retrieval())