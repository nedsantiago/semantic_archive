import semantic_archive
import logging

def test_test():
    LIST_SENTENCES: list[str] = [
        "Hydrologic & Hydraulic study of Bulacan City",
        "Community Engagement works in Manila City",
    ]
    archive = semantic_archive.SemanticArchive(LIST_SENTENCES)
    value = archive.search("Flood Control")
    assert value == "Should fail this"
