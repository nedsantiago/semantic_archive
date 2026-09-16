from dataclasses import dataclass

@dataclass
class Embedding:
    """
    Text and text-embedding pair. 
    Use `Embedding.text` to get the raw string. Use `Embedding.embedding` to
    get the embedding array.
    """

    embedding: list[float]
    text: str

