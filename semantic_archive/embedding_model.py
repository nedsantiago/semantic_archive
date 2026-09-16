import ollama
from google import genai
from .embedding_type import Embedding
import logging


logging.basicConfig(
    filename="data.log",
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

class EmbeddingModel:
    _valid_ollama_models: set[str] = {
        "qwen3-embedding",
        "embeddinggemma",
    }
    _valid_api_models: set[str] = {
        "gemini-embedding-2"
    }

    # set operation, union all valid embedding modle names
    _valid_models: set[str] = (
        _valid_ollama_models | _valid_api_models
    )

    def __init__(self, model: str, api_key: str = ""):
        if model not in self._valid_models:
            msg: str = (
                "Invalid embedding model. "
                f" \"{model}\" is not an available embedding model"
            )
            raise ValueError(msg)

        self._model: str = model
        self._api_key: str = api_key

        # setup a function that takes text and returns embedding
        if self._model in self._valid_ollama_models:
            logger.debug(f"Initializing a new Ollama Embedding model")
            def embed_func(text: str):
                return ollama.embed(
                    model=model,
                    input=text
                ).embeddings
            self._embedding_model = embed_func
        elif self._model in self._valid_api_models:
            if not api_key:
                raise ValueError(
                    f"EmbeddingModel {self._model} received no API Key"
                )
            logger.debug(f"Initializing a new Gemini Embedding model")

            # Initialize Google embedding model client
            client = genai.Client(api_key=api_key)
            def embed_func(text: str):
                # call the embedding model and parse the response
                # for the embedding
                return client.models.embed_content(
                    model=model,
                    contents=text
                ).embeddings[0].values
            self._embedding_model = embed_func
        else:
            msg: str = (
                "Invalid embedding model in EmbeddingModel. "
                f" \"{model}\" is not an available embedding model"
            )
            raise ValueError(msg)

        if not self._embedding_model:
            raise ValueError("Failed to make initialize an embedding model")

    def embed(self, text: str) -> Embedding:
        if type(text) is not str:
            msg: str = (
                "EmbeddingModel.embed only accepts strings, "
                f"received {type(text)}"
            )
            raise ValueError(msg)
        # do not accept empty strings
        if not text:
            raise ValueError(
                f"EmbeddingModel.embed received an empty string \"{text}\""
            )

        logger.debug(f"Creating a new Embedding object for text: {text}")
        return Embedding(
            text=text,
            embedding=self._embedding_model(text),
        )
