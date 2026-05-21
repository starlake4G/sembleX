from semble.config import RerankerConfig
from semble.interfaces import Reranker


def create_reranker(config: RerankerConfig | None = None) -> Reranker | None:
    if config is None:
        from semble.config import SembleConfig

        config = SembleConfig().reranker

    if config.backend == "cross_encoder":
        from semble.backends.reranker.cross_encoder import CrossEncoderReranker

        return CrossEncoderReranker(model_name=config.model or "BAAI/bge-reranker-v2-m3")

    if config.backend == "hybrid":
        from semble.backends.reranker.cross_encoder import CrossEncoderReranker
        from semble.backends.reranker.hybrid import HybridReranker
        from semble.backends.reranker.rules import RulesReranker

        rules = RulesReranker()
        model = CrossEncoderReranker(model_name=config.model or "BAAI/bge-reranker-v2-m3")
        return HybridReranker(rules=rules, model=model, coarse_multiplier=config.coarse_multiplier)

    from semble.backends.reranker.rules import RulesReranker

    return RulesReranker()
