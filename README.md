# Semantic Archive

A semantics-based search tool.

## Usage

```
from semantic_archive import SemanticArchive
from dotenv import load_dotenv
import os


load_dotenv()
API_KEY = os.getenv("GEMINI_EMBEDDING_2_KEY")

LIST_SENTENCES: list[str] = [
    "Hydrologic & Hydraulic study of Bulacan City",
    "Community Engagement works in Manila City",
]
MODEL: str = "gemini-embedding-2"
semantic_archive = SemanticArchive(LIST_SENTENCES, MODEL, API_KEY)

result = semantic_archive.search("Studying the hydrology of a city")
```

Only supports the following models: `gemini-embedding-2` (requires API keys), `qwen3-embedding` (requires ollama), `gemini-embedding-2` (requires ollama)
