from semantic_archive import SemanticArchive
from dotenv import load_dotenv
import os
import logging


load_dotenv()
API_KEY = os.getenv("GEMINI_EMBEDDING_2_KEY")

def test_SemanticArchive_runs():
    LIST_SENTENCES: list[str] = [
        "Hydrologic & Hydraulic study of Bulacan City",
        "Community Engagement works in Manila City",
    ]
    MODEL: str = "gemini-embedding-2"
    semantic_archive = SemanticArchive(LIST_SENTENCES, MODEL, API_KEY)

    result = semantic_archive.search("Studying the hydrology of a city")

    assert result[0][0] == "Hydrologic & Hydraulic study of Bulacan City"
