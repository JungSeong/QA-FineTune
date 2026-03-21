"""
sentence-transformers 기반 경량 임베딩 서버
OpenAI /v1/embeddings 호환 API

위치: QA-FineTune/main/embedding_server.py
실행: python3 embedding_server.py --port 8001 --model /embeddings/dragonkue/snowflake-arctic-embed-l-v2.0-ko
"""

import argparse
import time
import logging
from typing import List, Union

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("embedding_server")

# ─────────────────────────────────────────────────────────
# 스키마 (OpenAI 호환)
# ─────────────────────────────────────────────────────────

class EmbeddingRequest(BaseModel):
    input:           Union[str, List[str]]
    model:           str = "snowflake-arctic-embed-l-v2.0-ko"
    encoding_format: str = "float"


class EmbeddingObject(BaseModel):
    object:    str = "embedding"
    index:     int
    embedding: List[float]


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data:   List[EmbeddingObject]
    model:  str
    usage:  dict


class ModelObject(BaseModel):
    id:         str
    object:     str = "model"
    created:    int
    owned_by:   str = "sentence-transformers"


class ModelListResponse(BaseModel):
    object: str = "list"
    data:   List[ModelObject]


# ─────────────────────────────────────────────────────────
# 전역 상태
# ─────────────────────────────────────────────────────────

model:      SentenceTransformer = None
model_name: str                 = ""


# ─────────────────────────────────────────────────────────
# FastAPI 앱
# ─────────────────────────────────────────────────────────

app = FastAPI(title="Embedding Server (OpenAI Compatible)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/v1/models", response_model=ModelListResponse)
def list_models():
    return ModelListResponse(
        data=[
            ModelObject(
                id=model_name,
                created=int(time.time()),
            )
        ]
    )


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
def create_embeddings(req: EmbeddingRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="모델이 로드되지 않았습니다.")

    texts = [req.input] if isinstance(req.input, str) else req.input

    try:
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
    except Exception as e:
        logger.error("임베딩 생성 실패: %s", e)
        raise HTTPException(status_code=500, detail=f"임베딩 오류: {e}")

    total_tokens = sum(len(t.split()) for t in texts)

    return EmbeddingResponse(
        data=[
            EmbeddingObject(index=i, embedding=emb.tolist())
            for i, emb in enumerate(embeddings)
        ],
        model=model_name,
        usage={"prompt_tokens": total_tokens, "total_tokens": total_tokens},
    )


# ─────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────

def main():
    global model, model_name

    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  type=str, default="/embeddings/dragonkue/snowflake-arctic-embed-l-v2.0-ko")
    parser.add_argument("--port",   type=int, default=8001)
    parser.add_argument("--host",   type=str, default="0.0.0.0")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model_name = args.model.rstrip("/").split("/")[-1]

    logger.info("🚀 임베딩 모델 로딩 중: %s (device: %s)", args.model, args.device)
    model = SentenceTransformer(args.model, device=args.device)
    logger.info("✅ 임베딩 모델 로드 완료: %s", model_name)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()