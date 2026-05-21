from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np
import numpy.typing as npt
from huggingface_hub.utils.tqdm import disable_progress_bars
from model2vec import StaticModel

from semble.interfaces import EmbeddingProvider, EmbeddingMatrix


class Model2VecEmbedding(EmbeddingProvider):
    def __init__(self, model_path: str = "minishlab/potion-code-16M") -> None:
        disable_progress_bars()
        try:
            self._model = StaticModel.from_pretrained(model_path)
        finally:
            disable_progress_bars()

    @property
    def name(self) -> str:
        return "model2vec"

    @property
    def dim(self) -> int:
        return self._model.dim

    def encode(self, texts: Sequence[str]) -> EmbeddingMatrix:
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        return np.array(
            self._model.encode(list(texts), use_multiprocessing=False),
            dtype=np.float32,
        )
