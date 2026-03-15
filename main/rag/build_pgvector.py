"""
document를 vectorstore에 저장해주는 함수

[pgvector 초기화]
docker exec -it postgres psql -U postgres -d postgres
SELECT name FROM langchain_pg_collection;
DELETE FROM langchain_pg_embedding
WHERE collection_id = (
    SELECT uuid FROM langchain_pg_collection
    WHERE name = 'library-qa'
);
DELETE FROM langchain_pg_collection
WHERE name = 'library-qa';

[pgvector에 새로운 엑셀 데이터 추가]
QA-FineTune/docker에서
docker compose up -d
이후 파이썬 파일 실행
"""

import pandas as pd
import os
from langchain_postgres import PGVector
from langchain_community.document_loaders import DataFrameLoader
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).parent.parent.parent / "docker" / ".env")

DB_PASSWORD    = os.getenv("POSTGRES_PASSWORD")
DB_USER        = os.getenv("POSTGRES_USER")
DB_NAME        = os.getenv("POSTGRES_DB")
DB_HOST        = "localhost"
EMBEDDING_API_URL = "http://localhost:8003/v1"

print("🚀 데이터 로딩 시작...")
# df = pd.read_excel("../data/raw/고양시도서관 FAQ1.xlsx")
df = pd.read_excel("../data/raw/강원도교육청도서관 FAQ1.xlsx")
df["content"] = df["TITLE"].str.strip() + "\n" + df["DES"].str.strip()

loader = DataFrameLoader(df, page_content_column="content")
docs   = loader.load()
print(f"✅ {len(docs)}개 문서 로드 완료")

embeddings = OpenAIEmbeddings(
    model="snowflake-arctic-embed-l-v2.0-ko",
    openai_api_base=EMBEDDING_API_URL,
    openai_api_key="none",
)

connection_uri = f"postgresql+psycopg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:5432/{DB_NAME}"

print("📦 PGVector에 저장 중...")
vectorstore = PGVector.from_documents(
    documents=docs,
    embedding=embeddings,
    collection_name="library-qa",
    connection=connection_uri,   # ← engine 대신 URI 문자열 직접 전달
    pre_delete_collection=True,
)

print("✅ Successfully saved to PGVector!!")