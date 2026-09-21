# import functions_framework
from semantic_archive import SemanticArchive
import os


API_KEY = os.getenv("GEMINI_EMBEDDING_2_KEY")

# @functions_framework.http
def main(request):
    LIST_SENTENCES: list [str] = [
    ]
    MODEL: str = "gemini-embedding-2"


    semantic_archive = SemanticArchive(
        LIST_SENTENCES,
        MODEL,
        API_KEY,
    )

    request_json = request.get_json(silent=True)
    request_args = request.args

    if request_json and "text" in request_json:
        text = request_json["text"]
    elif request_args and "text" in request_args:
        text = request_args["text"]
    else:
        text = ""

    result = semantic_archive.search(text, threshold=0.60)

    return f"{result}"
