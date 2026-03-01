import pandas as pd
import csv
import os
import urllib.parse
from langchain_postgres import PGVector
from langchain_community.document_loaders import DataFrameLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

db_password = os.getenv('db_password')

print("🚀 데이터 로딩 시작...")
df = pd.read_excel("../../data/고양시도서관 FAQ1.xlsx")
loader = DataFrameLoader(df, page_content_column="DES") # page_content_column에 있는 텍스트를 실제로 숫자로 변환하여 저장
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=300)
splits = text_splitter.split_documents(docs)

print(f"✂️ 총 {len(splits)}개의 청크로 분할되었습니다.")
print(f"Example : {splits[0]}")

embeddings = HuggingFaceEmbeddings(
    model_name="jhgan/ko-sroberta-nli",
    model_kwargs={'device': 'cuda'}
)

print(f"📦 PGVector에 저장 중...")

vectorstore = PGVector.from_documents(
    documents=splits,
    embedding=embeddings,
    collection_name="library-qa", # 같은 collection_name -> 해당 이름 아래에만 문서가 저장됨
    connection=f"postgresql+psycopg://postgres:{db_password}@localhost:5432/postgres",
    pre_delete_collection=True # 기존 데이터 지우고 새로 시작
)

print(f"Successfully saved to PGVector!!")