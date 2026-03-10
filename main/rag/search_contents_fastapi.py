import os
from fastapi import FastAPI
from sqlalchemy.ext.asyncio import create_async_engine
from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings # vLLM은 OpenAI 규격을 따릅니다.

app = FastAPI()

# 1. 환경 변수 또는 설정
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD")
DB_USER = os.getenv("POSTGRES_USER")
DB_NAME = os.getenv("POSTGRES_DB")
DB_HOST = "postgres"

EMBEDDING_API_URL = "http://exaone:8001/v1" 

# 2. 임베딩 모델 설정 (로컬 로드 대신 API 호출)
embeddings = OpenAIEmbeddings(
    model="snowflake-arctic-embed-l-v2.0-ko", # vLLM에 설정된 모델명과 동일해야 함
    openai_api_base=EMBEDDING_API_URL,
    openai_api_key="none" # vLLM은 키가 필요 없지만 라이브러리 규격상 아무 값이나 입력
)

connection_uri = f"postgresql+psycopg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:5432/{DB_NAME}"
engine = create_async_engine(connection_uri)

# 3. Vectorstore 연결
vectorstore = PGVector(
    embeddings=embeddings,
    collection_name="library-qa",
    connection=engine
)

@app.get("/search")
async def search(query: str, k: int = 3):
    print(f"🔍 검색 요청 수신: {query}")
    try:
        # 이 과정에서 내부적으로 exaone:8001 서버에 임베딩을 요청합니다.
        docs = await vectorstore.asimilarity_search(query, k=k)
        print(f"✅ 검색 결과 개수: {len(docs)}")
        return {"results": [doc.page_content for doc in docs]}
    except Exception as e:
        print(f"❌ 검색 에러: {str(e)}")
        return {"error": str(e), "results": []}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=4200)