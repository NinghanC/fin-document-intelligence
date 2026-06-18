__all__ = ["VectorStoreService", "KnowledgeGraphService", "MultimodalService"]


def __getattr__(name):
    if name == "VectorStoreService":
        from .vector_store import VectorStoreService

        return VectorStoreService
    if name == "KnowledgeGraphService":
        from .knowledge_graph import KnowledgeGraphService

        return KnowledgeGraphService
    if name == "MultimodalService":
        from .multimodal import MultimodalService

        return MultimodalService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
