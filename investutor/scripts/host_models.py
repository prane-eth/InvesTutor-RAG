#!/usr/bin/env python3

# Hosting models through an API
# Why host the models?
# - When we restart the evaluator or the server, we should not wait for the models to load repeatedly.
# - We can host the models on a server and access them through an API.

import os
import time
from datetime import datetime
from typing import List, Tuple

import psutil
import pytz
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder, SentenceTransformer
from functools import lru_cache


load_dotenv()

app = FastAPI()

device = "cuda"  # cpu/cuda


print("Loading models...")


# ------------------ Reranking Model ------------------

print("Loading the ranking model...")
RERANK_MODEL_NAME = os.getenv("RERANK_MODEL_NAME", "")
if not RERANK_MODEL_NAME:
    raise Exception("RERANK_MODEL_NAME is not set in the environment variables")

ranker = CrossEncoder(RERANK_MODEL_NAME, device=device)
ranker.predict([("hello", "world")])


class RerankRequest(BaseModel):
    query: str
    documents: List[str]
    top_n: int


@lru_cache(maxsize=50000)
def rerank_results_cached(query: str, documents: Tuple[str], top_k: int):
    return ranker.rank(query, list(documents), top_k, return_documents=True)


@app.post("/v2/rerank")  # Cohere-compatible endpoint
def rerank_results(req: RerankRequest):
    # To get a list of relevant documents based on a query.
    results = rerank_results_cached(req.query, tuple(req.documents), req.top_n)
    new_results = []
    for result in results:
        new_results.append(
            {
                "index": result["corpus_id"],
                "relevance_score": float(result["score"]),
                "text": result["text"],
            }
        )
    return {"results": new_results}


# ------------------ Embedding Model ------------------

print("Loading the embedding model...")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "")
if not EMBED_MODEL_NAME:
    raise Exception("EMBED_MODEL_NAME is not set in the environment variables")

embed_model = SentenceTransformer(EMBED_MODEL_NAME, device=device)
embed_model.encode(["hello world"])


class EmbedRequest(BaseModel):
    # alias the Pydantic field `inputs` to the JSON key 'input'
    inputs: List[str] = Field(..., alias="input")

    class Config:
        validate_by_name = True


@lru_cache(maxsize=50000)
def embed_texts_cached(texts: Tuple[str]):
    return embed_model.encode(
        list(texts), batch_size=64, convert_to_numpy=True, normalize_embeddings=False
    )


@app.post("/v1/embeddings")
def embed_texts(req: EmbedRequest):  # OpenAI-compatible endpoint
    # To embed texts using the sentence-transformer model.
    embeddings = embed_texts_cached(tuple(req.inputs))
    return {
        "data": [
            {"index": idx, "object": "embedding", "embedding": embedding.tolist()}
            for idx, embedding in enumerate(embeddings)
        ]
    }


# ------------------ Health Check ------------------

ist = pytz.timezone("Asia/Kolkata")


def get_memory_usage():
    used = psutil.virtual_memory().used / (1024 * 1024)  # Convert bytes to MB
    total = psutil.virtual_memory().total / (1024 * 1024)  # Convert bytes to MB
    return f"{used:.2f} MB / {total:.2f} MB ({used / total * 100:.2f}%)"


@app.get("/")
@app.get("/ping")
@app.get("/health")
def root_test():
    current_time = datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S %p %Z")
    return {
        "message": "Model hosting service is running at " + str(current_time),
        "Memory usage": get_memory_usage(),
    }


if __name__ == "__main__":
    base_url = os.getenv("RERANK_BASE_URL", "")
    if not base_url:
        raise ValueError("RERANK_BASE_URL environment variable is not set.")

    # get port from the base_url
    url_parts = base_url.split(":")
    port = int(url_parts[-1].split("/")[0]) if len(url_parts) > 2 else 8000

    while True:
        try:
            # Start the FastAPI server
            print("Starting the FastAPI server...")
            uvicorn.run(app, host="0.0.0.0", port=port)
            # $ uvicorn investutor.scripts.host_models:app --host 0.0.0.0 --port 8000 --workers 2
            break
        except KeyboardInterrupt:
            print("Server stopped by user.")
            break
        except Exception as e:
            print(f"Error starting the server: {e}.")
            print("Retrying in 5 seconds...")
            time.sleep(5)
