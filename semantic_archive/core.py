from .embedding_type import Embedding
from .embedding_model import EmbeddingModel
import numpy as np
import logging


logging.basicConfig(
    filename="data.log",
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

class SemanticArchive:
    def __init__(
            self, sentences: list, model_code:str,
            api_key: str = ""):

        logger.info(
            f"Initializing SemanticArchive with {len(sentences)} sentences"
        )
        
        self._embedding_model: EmbeddingModel = EmbeddingModel(
            model_code,
            api_key=api_key,
        )
        self._archive: list[Embedding] = []
        for s in sentences:
            self._archive.append(self._embedding_model.embed(s))

    def search(self, search_text: str, threshold=0.65) -> list[(str, float)]:
        # when given anything other than string
        dtype = type(search_text)
        if dtype is not str:
            raise TypeError(
                f"SemanticArchive.search needs string, received {dtype}"
            )

        # return an empty list when given an empty string
        if not search_text:
            return []

        logger.info(f"SemanticArchive searches for \"{search_text}\"")

        search_embedding: Embedding = self._embedding_model.embed(
            search_text,
        )

        similar_sentences: list[str] = []
        for embd in self._archive:
            arr1 = search_embedding.embedding
            arr2 = embd.embedding

            # Cosine Similarity Function
            cosine_similarity: float = _calc_cosine_similarity(arr1, arr2)

            if cosine_similarity > threshold:
                # Append the matching text and similarity score
                similar_sentences.append((embd.text, cosine_similarity))

        # sort from best to worst similarity score
        return sorted(similar_sentences, key= lambda x: -x[1])

def _calc_cosine_similarity(arr1, arr2) -> float:
    return np.dot(
        arr1, arr2
    ) / (np.linalg.norm(arr1) * np.linalg.norm(arr2))
