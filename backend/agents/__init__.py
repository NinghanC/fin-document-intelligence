__all__ = [
    "DocParserAgent",
    "KnowledgeExtractAgent",
    "QAAgent",
    "KnowledgeUpdateAgent",
]


def __getattr__(name):
    if name == "DocParserAgent":
        from .doc_parser_agent import DocParserAgent

        return DocParserAgent
    if name == "KnowledgeExtractAgent":
        from .knowledge_extract_agent import KnowledgeExtractAgent

        return KnowledgeExtractAgent
    if name == "KnowledgeUpdateAgent":
        from .knowledge_update_agent import KnowledgeUpdateAgent

        return KnowledgeUpdateAgent
    if name == "QAAgent":
        from .qa_agent import QAAgent

        return QAAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
