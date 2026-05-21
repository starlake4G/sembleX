from semble.config import VectorStoreConfig
from semble.interfaces import VectorStore


def create_vector_store(config: VectorStoreConfig | None = None, dim: int = 256) -> VectorStore:
    """Instantiate the local-side dense vector store described by *config*."""
    if config is None:
        from semble.config import SembleConfig

        config = SembleConfig().vector_store

    if config.backend == "faiss":
        from semble.backends.vector_store.faiss_store import FaissVectorStore

        return FaissVectorStore(
            dim=dim,
            use_gpu=config.use_gpu,
            index_type=config.index_type,
            metric=config.metric,
            nlist=config.ivf_nlist,
        )

    from semble.backends.vector_store.numpy_store import NumpyVectorStore

    return NumpyVectorStore(dim=dim)
