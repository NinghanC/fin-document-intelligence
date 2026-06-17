"""
Application settings - loaded from environment variables or a .env file
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LLM
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o"
    embedding_model: str = "text-embedding-3-small"
    embedding_provider: str = "auto"  # auto | openai | local | hash

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "local-password"

    # Vector Store
    vector_store_type: str = "chroma"  # chroma | pgvector
    chroma_mode: str = "local"  # local | http
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    pgvector_dsn: str = "postgresql://postgres:postgres@localhost:5432/knowledge"
    disable_local_embeddings: bool = False

    # Kafka (CDC)
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_topic_doc_changes: str = "doc-changes"
    kafka_topic_kg_updates: str = "kg-updates"
    enable_cdc_consumer: bool = False

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8080
    api_key: str = ""
    auth_enabled: bool = False
    allowed_origins: str = "http://localhost:8080,http://127.0.0.1:8080"
    rate_limit_requests: int = 60
    rate_limit_window_seconds: int = 60
    max_upload_size_mb: int = 10

    # Document Store
    upload_dir: str = "./uploads"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
