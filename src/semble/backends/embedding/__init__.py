from semble.config import EmbeddingConfig
from semble.interfaces import EmbeddingProvider


def create_embedding_provider(config: EmbeddingConfig | None = None) -> EmbeddingProvider:
    """Instantiate the OpenAI-compatible embedding provider described by *config*."""
    if config is None:
        from semble.config import SembleConfig

        config = SembleConfig().embedding

    from semble.backends.embedding.openai_compat import OpenAICompatEmbedding

    return OpenAICompatEmbedding(
        base_url=config.openai_base_url,
        api_key=config.openai_api_key or "",
        model=config.openai_model,
        dim=config.openai_dim,
        batch_size=config.batch_size,
        max_retries=config.max_retries,
        max_concurrent=config.max_concurrent,
        max_context_tokens=config.max_context_tokens,
    )
