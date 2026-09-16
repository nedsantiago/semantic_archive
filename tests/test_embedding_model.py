from semantic_archive.embedding_model import EmbeddingModel
import os
from dotenv import load_dotenv
import logging


load_dotenv()
api_key = os.getenv("GEMINI_EMBEDDING_2_KEY")

def test_gemini_embedding_model_runs():
    embedding_model = EmbeddingModel(
        "gemini-embedding-2",
        api_key=api_key,
    )
    embedding = embedding_model.embed("The sky is blue")
    assert embedding
