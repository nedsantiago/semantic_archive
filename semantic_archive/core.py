import ollama
from dataclasses import dataclass
import numpy as np
import logging


logging.basicConfig(
    filename="data.log",
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

@dataclass
class Embedding:
    """
    Text and text-embedding pair. 
    Use `Embedding.text` to get the raw string. Use `Embedding.embedding` to
    get the embedding array.
    """

    embedding: list[float]
    text: str

def calc_embedding(text: str, model: str="qwen3-embedding") -> Embedding:
    if type(text) is not str:
        raise ValueError(
            f"calc_embedding only accepts strings, received {type(text)}"
        )
    # do not accept empty strings
    if not text:
        raise ValueError(
            f"calc_embedding received an empty string \"{text}\""
        )

    match model:
        case "qwen3-embedding":
            embeddings = ollama.embed(
                model=model,
                input=text,
            ).embeddings
        case "embeddinggemma":
            embeddings = ollama.embed(
                model=model,
                input=text,
            ).embeddings
        case _:
            msg: str = (
                "Invalid embedding model in calc_embedding. "
                f" \"{model}\" is not an available embedding model"
            )
            raise ValueError(msg)

    logger.debug(f"Creating a new Embedding object for text: {text}")

    return Embedding(
        text=text,
        embedding=embeddings,
    )

class SemanticArchive:
    def __init__(self, sentences: list):
        logger.info(
            f"Initializing SemanticArchive with {len(sentences)} sentences"
        )
        
        self._archive: list[Embedding] = []
        for s in sentences:
            self._archive.append(calc_embedding(s, model="embeddinggemma"))

    def search(self, search_text: str, threshold=0.65) -> list[(str, float)]:
        # return an empty list when searching for None, 0, or empty string
        if not search_text:
            return []

        # when given anything other than string
        dtype = type(search_text)
        if dtype is not str:
            raise TypeError(
                f"SemanticArchive.search needs string, received {dtype}"
            )

        logger.info(f"SemanticArchive searches for \"{search_text}\"")

        search_embedding: Embedding = calc_embedding(
            search_text,
            model="embeddinggemma",
        )

        similar_sentences: list[str] = []
        for embd in self._archive:
            arr1 = search_embedding.embedding
            arr2 = embd.embedding

            # Cosine Similarity Function
            cosine_similarity: float = np.dot(
                arr1,
                arr2
            ) / (np.linalg.norm(arr1) * np.linalg.norm(arr2))

            if cosine_similarity > threshold:
                # Append the matching text and similarity score
                similar_sentences.append((embd.text, cosine_similarity))

        # sort from best to worst similarity score
        return sorted(similar_sentences, key= lambda x: -x[1])
