import os
import sys
from mcp.server.fastmcp import FastMCP
from langchain_postgres import PGVector
from langchain_huggingface import HuggingFaceEmbeddings

MCP_PORT = int(os.getenv("MCP_PORT", "8080"))
mcp = FastMCP("Library-Search", host="0.0.0.0", port=MCP_PORT)

DB_PASSWORD = "Passw0rd!"
DB_HOST = "postgres" 

# 2. 임베딩 모델 설정 - device를 'cpu'로 변경
embeddings = HuggingFaceEmbeddings(
    model_name="jhgan/ko-sroberta-nli",
    model_kwargs={'device': 'cpu'} # 🌟 'cuda'에서 'cpu'로 변경
)

# 3. Vectorstore 연결
vectorstore = PGVector(
    embeddings=embeddings,
    collection_name="library-qa",
    connection=f"postgresql+psycopg://postgres:{DB_PASSWORD}@{DB_HOST}:5432/postgres",
)

@mcp.tool()
def search_library(query: str, k: int = 3) -> str:
    """
    도서관 이용 규정, 상호대차 서비스 등과 관련된 문서 본문을 검색합니다.
    질문(query)을 입력하면 가장 관련 있는 문맥 3개를 합쳐서 반환합니다.
    """
    try:
        docs = vectorstore.similarity_search(query, k=k)
        context = "\n\n".join([doc.page_content for doc in docs])
        return context
    except Exception as e:
        return f"검색 중 오류가 발생했습니다: {str(e)}"

if __name__ == "__main__":
    print(f"🚀 PostgreSQL MCP 서버 시작 (Port: {MCP_PORT})")
    mcp.run(transport="sse")