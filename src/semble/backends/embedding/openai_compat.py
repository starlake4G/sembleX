from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from semble.interfaces import EmbeddingFailure, EmbeddingMatrix, EmbeddingProvider

logger = logging.getLogger(__name__)

# Headroom kept below the model's hard context limit. The local token estimate
# (tiktoken cl100k_base, or a chars-based heuristic) rarely matches the serving
# tokenizer exactly, and the server may add special tokens (BOS/EOS). Without a
# margin, a text estimated at exactly the limit can overflow by a few tokens and
# get rejected with HTTP 400. 5% + a small fixed reserve absorbs that drift.
_CONTEXT_SAFETY_RATIO = 0.95
_CONTEXT_RESERVE_TOKENS = 16


def _token_counter(model: str) -> Callable[[str], int]:
    try:
        import tiktoken
    except ImportError:
        return lambda text: max(1, len(text) // 3)
    try:
        enc = tiktoken.encoding_for_model(model)
    except (KeyError, ValueError):
        enc = tiktoken.get_encoding("cl100k_base")
    return lambda text: len(enc.encode(text, disallowed_special=()))


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
        # Effective truncation budget: stay safely under the model's hard limit.
        self._token_budget = max(1, int(max_context_tokens * _CONTEXT_SAFETY_RATIO) - _CONTEXT_RESERVE_TOKENS)
        self._count_tokens = _token_counter(model)
        self._dim_lock = threading.Lock()

    @property
    def name(self) -> str:
        return "openai_compat"

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts: Sequence[str]) -> EmbeddingMatrix:
        """Embed *texts*. Raises EmbeddingFailure if any text cannot be encoded."""
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
            batches = [(i, text_list[i : i + self._batch_size]) for i in range(0, total, self._batch_size)]
            logger.info(
                "Encoding %d texts in %d batches (batch_size=%d, workers=%d)...",
                total,
                len(batches),
                self._batch_size,
                self._max_concurrent,
            )
            done_count = 0
            with ThreadPoolExecutor(max_workers=self._max_concurrent) as executor:
                futures = {executor.submit(self._encode_batch, batch): start_idx for start_idx, batch in batches}
                for future in as_completed(futures):
                    start_idx = futures[future]
                    try:
                        batch_result = future.result()
                    except Exception as exc:
                        logger.error("Batch starting at %d raised: %s", start_idx, exc)
                        batch_result = [None] * len(batches[start_idx // self._batch_size][1])
                    for i, emb in enumerate(batch_result):
                        all_embeddings[start_idx + i] = emb
                    done_count += 1
                    logger.info("Encoding progress: %d/%d batches done", done_count, len(batches))

        failed_indices = [i for i, emb in enumerate(all_embeddings) if emb is None]
        successes = [emb for emb in all_embeddings if emb is not None]

        if not successes:
            raise EmbeddingFailure(
                f"All {total} embedding requests failed",
                failed_indices=failed_indices,
            )

        actual_dim = len(successes[0])
        if actual_dim != self._dim:
            with self._dim_lock:
                self._dim = actual_dim
                logger.info("Updated embedding dimension to %d", self._dim)

        if failed_indices:
            partial = self._normalize(np.array(successes, dtype=np.float32))
            logger.warning(
                "Embedding partial failure: %d/%d texts failed; surfacing partial result",
                len(failed_indices),
                total,
            )
            raise EmbeddingFailure(
                f"{len(failed_indices)} of {total} embeddings failed",
                failed_indices=failed_indices,
                partial=partial,
            )

        full = np.array(all_embeddings, dtype=np.float32)
        return self._normalize(full)

    @staticmethod
    def _normalize(arr: np.ndarray) -> EmbeddingMatrix:
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.where(norms > 1e-9, norms, 1.0)
        return (arr / norms).astype(np.float32, copy=False)

    def _truncate(self, text: str) -> str:
        budget = self._token_budget
        if self._count_tokens(text) <= budget:
            return text
        lo, hi = 1, len(text)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self._count_tokens(text[:mid]) <= budget:
                lo = mid
            else:
                hi = mid - 1
        logger.warning(
            "Truncated oversized input from ~%d to ~%d tokens (budget %d, model limit %d)",
            self._count_tokens(text),
            self._count_tokens(text[:lo]),
            budget,
            self._max_context_tokens,
        )
        return text[:lo]

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
            except Exception as exc:  # noqa: BLE001 — provider-defined exception hierarchies vary
                if attempt == self._max_retries - 1:
                    logger.error("Single encode failed after %d retries: %s", self._max_retries, exc)
                    return None
                wait = 2**attempt
                logger.warning("Retry %d/%d single: %s (waiting %ds)", attempt + 1, self._max_retries, exc, wait)
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
            except Exception as exc:  # noqa: BLE001
                if attempt == self._max_retries - 1:
                    logger.warning("Batch failed, falling back to single encoding: %s", exc)
                    return [self._encode_single(t) for t in truncated]
                wait = 2**attempt
                logger.warning("Retry %d/%d after error: %s (waiting %ds)", attempt + 1, self._max_retries, exc, wait)
                time.sleep(wait)
        return [None] * len(texts)
