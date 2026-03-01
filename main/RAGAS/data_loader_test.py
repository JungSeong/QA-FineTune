import pandas as pd
from langchain_community.document_loaders import DataFrameLoader

df = pd.read_excel("/home/vsc/LLM_TUNE/QA-FineTune/main/data/고양시도서관 FAQ1.xlsx")

df['combined_text'] = df.apply(
    lambda row: f"FAQ: {row['FAQ']}\TITLE: {row['TITLE']}\nDES: {row['DES']}", 
    axis=1
)

loader = DataFrameLoader(df, page_content_column="combined_text")

# 문서 로드
docs = loader.load()

# 문서 길이 출력
print(len(docs))
print(docs[0])