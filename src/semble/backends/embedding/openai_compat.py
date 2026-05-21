from __future__ import annotations

import logging
import threading
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import numpy.typing as npt

from semble.interfaces import EmbeddingProvider, EmbeddingMatrix

logger = logging.getLogger(__name__)


class OpenAICompatEmbedding(EmbeddingProvider):
    def __init__(
        self,
        base_url: str = "https://api.openai.com/v1",
        api_key: str = "",
        model: str = "text-embedding-3-small",
        dim: int = 1536,
        batch_size: int = 2048,
        max_retries: int = 3,
        max_concurrent: int = 10,
        max_context_tokens: int = 8192,
    ) -> None:
        from openai import OpenAI

        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._model = model
        self._dim = dim
        self._batch_size = batch_size
        self._max_retries = max_retries
        self._max_concurrent = max_concurrent
        self._max_context_tokens = max_context_tokens
        self._max_chars = max_context_tokens * 3
        self._dim_lock = threading.Lock()

    @property
    def name(self) -> str:
        return "openai_compat"

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts: Sequence[str]) -> EmbeddingMatrix:
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)

        text_list = list(texts)
        total = len(text_list)
        all_embeddings: list[list[float] | None] = [None] * total

        if total <= self._batch_size:
            logger.info("Encoding %d texts in 1 batch...", total)
            batch_result = self._encode_batch(text_list)
            for i, emb in enumerate(batch_result):
                all_embeddings[i] = emb
        else:
            batches = [
                (i, text_list[i : i + self._batch_size])
                for i in range(0, total, self._batch_size)
            ]
            logger.info("Encoding %d texts in %d batches (batch_size=%d, workers=%d)...", total, len(batches), self._batch_size, self._max_concurrent)
            done_count = 0
            with ThreadPoolExecutor(max_workers=self._max_concurrent) as executor:
                futures = {
                    executor.submit(self._encode_batch, batch): start_idx
                    for start_idx, batch in batches
                }
                for future in as_completed(futures):
                    start_idx = futures[future]
                    try:
                        batch_result = future.result()
                        for i, emb in enumerate(batch_result):
                            all_embeddings[start_idx + i] = emb
                    except Exception as e:
                        logger.error("Failed to encode batch starting at %d: %s", start_idx, e)
                    done_count += 1
                    logger.info("Encoding progress: %d/%d batches done", done_count, len(batches))

        failed_indices = [i for i, e in enumerate(all_embeddings) if e is None]
        if len(failed_indices) == total:
            raise RuntimeError("All embedding requests failed")

        actual_dim = next((len(e) for e in all_embeddings if e is not None), self._dim)
        if actual_dim != self._dim:
            with self._dim_lock:
                self._dim = actual_dim
                logger.info("Updated embedding dimension to %d", self._dim)

        for i in failed_indices:
            all_embeddings[i] = [0.0] * self._dim

        arr = np.array(all_embeddings, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.where(norms > 1e-9, norms, 1.0)
        arr = arr / norms
        if failed_indices:
            arr[failed_indices] = 0.0
        return arr

    def _truncate(self, text: str) -> str:
        if len(text) <= self._max_chars:
            return text
        return text[: self._max_chars]

    def _encode_single(self, text: str) -> list[float] | None:
        text = self._truncate(text)
        for attempt in range(self._max_retries):
            try:
                response = self._client.embeddings.create(
                    model=self._model,
                    input=[text],
                    encoding_format="float",
                )
                return response.data[0].embedding
            except Exception as e:
                if attempt == self._max_retries - 1:
                    logger.error("Single encode failed after %d retries: %s", self._max_retries, e)
                    return None
                wait = 2**attempt
                logger.warning("Retry %d/%d single: %s (waiting %ds)", attempt + 1, self._max_retries, e, wait)
                time.sleep(wait)
        return None

    def _encode_batch(self, texts: list[str]) -> list[list[float] | None]:
        truncated = [self._truncate(t) for t in texts]
        for attempt in range(self._max_retries):
            try:
                response = self._client.embeddings.create(
                    model=self._model,
                    input=truncated,
                    encoding_format="float",
                )
                sorted_data = sorted(response.data, key=lambda x: x.index)
                return [d.embedding for d in sorted_data]
            except Exception as e:
                if attempt == self._max_retries - 1:
                    logger.warning("Batch failed, falling back to single encoding: %s", e)
                    results: list[list[float] | None] = []
                    for t in truncated:
                        results.append(self._encode_single(t))
                    return results
                wait = 2**attempt
                logger.warning("Retry %d/%d after error: %s (waiting %ds)", attempt + 1, self._max_retries, e, wait)
                time.sleep(wait)
        return [None] * len(texts)
