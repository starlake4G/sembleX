from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

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
# When the server still rejects a truncated input (tokenizer mismatch), shrink the
# budget by this factor and retry, down to this floor — guarantees we never drop a chunk.
_OVERFLOW_SHRINK = 0.8
_MIN_BUDGET = 256
_MAX_SHRINKS = 8


def _exact_token_counter(tokenizer: str) -> Callable[[str], int] | None:
    """Build a token counter from the serving model's real tokenizer, if loadable.

    *tokenizer* may be a local path (a ``tokenizer.json`` file or a directory holding
    one) or a Hugging Face / modelscope model id. Tries, in order: the lightweight
    ``tokenizers`` library (no model weights), then ``transformers``, then
    ``modelscope``. Returns ``None`` if none can resolve it, so the caller can fall
    back to an approximate counter (the encode path still retries on overflow).
    """
    # 1. Lightweight `tokenizers` library — handles a local tokenizer.json or a hub id.
    try:
        from tokenizers import Tokenizer

        path = Path(tokenizer)
        json_path = path / "tokenizer.json" if path.is_dir() else path
        tok = Tokenizer.from_file(str(json_path)) if json_path.is_file() else Tokenizer.from_pretrained(tokenizer)
        logger.info("Using exact tokenizer %r (tokenizers) for token counting", tokenizer)
        return lambda text: len(tok.encode(text).ids)
    except Exception:  # noqa: BLE001 — fall through to transformers / modelscope.
        logger.debug("tokenizers could not load %r", tokenizer, exc_info=True)

    # 2/3. transformers or modelscope AutoTokenizer (load tokenizer only, not weights).
    auto_loaders = []
    try:
        from transformers import AutoTokenizer as HFAutoTokenizer

        auto_loaders.append(HFAutoTokenizer)
    except ImportError:
        pass
    try:
        from modelscope import AutoTokenizer as MSAutoTokenizer

        auto_loaders.append(MSAutoTokenizer)
    except ImportError:
        pass
    for loader in auto_loaders:
        try:
            auto = loader.from_pretrained(tokenizer, trust_remote_code=True)
        except Exception:  # noqa: BLE001 — try the next loader / fall back.
            logger.warning("Could not load tokenizer %r via %s", tokenizer, loader.__module__, exc_info=True)
            continue
        logger.info("Using exact tokenizer %r (%s) for token counting", tokenizer, loader.__module__)
        return lambda text: len(auto.encode(text))
    return None


def _approx_token_counter(model: str) -> Callable[[str], int]:
    try:
        import tiktoken
    except ImportError:
        return lambda text: max(1, len(text) // 3)
    try:
        enc = tiktoken.encoding_for_model(model)
    except (KeyError, ValueError):
        enc = tiktoken.get_encoding("cl100k_base")
    return lambda text: len(enc.encode(text, disallowed_special=()))


def _is_context_overflow(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "maximum context length" in msg or "input_tokens" in msg or "context length" in msg


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
        tokenizer: str | None = None,
    ) -> None:
        from openai import OpenAI

        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._model = model
        self._dim = dim
        self._batch_size = batch_size
        self._max_retries = max_retries
        self._max_concurrent = max_concurrent
        self._max_context_tokens = max_context_tokens
        exact = _exact_token_counter(tokenizer) if tokenizer else None
        self._exact_tokenizer = exact is not None
        self._count_tokens = exact or _approx_token_counter(model)
        # Effective truncation budget: stay safely under the model's hard limit. With an
        # exact tokenizer the margin is just for server-side special tokens; with an
        # approximate counter it also absorbs tokenizer drift.
        ratio = 0.99 if self._exact_tokenizer else _CONTEXT_SAFETY_RATIO
        self._token_budget = max(1, int(max_context_tokens * ratio) - _CONTEXT_RESERVE_TOKENS)
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

    def _truncate(self, text: str, budget: int | None = None) -> str:
        budget = self._token_budget if budget is None else budget
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
        budget = self._token_budget
        payload = self._truncate(text, budget)
        attempt = 0
        shrinks = 0
        while True:
            try:
                response = self._client.embeddings.create(
                    model=self._model,
                    input=[payload],
                    encoding_format="float",
                )
                return response.data[0].embedding
            except Exception as exc:  # noqa: BLE001 — provider-defined exception hierarchies vary
                # The local token estimate can disagree with the server's tokenizer; if the
                # server says the input is still too long, shrink the budget and retry instead
                # of dropping the chunk. Shrink-retries are bounded separately from the
                # transient-error retries so they don't consume that budget.
                if _is_context_overflow(exc) and budget > _MIN_BUDGET and shrinks < _MAX_SHRINKS:
                    shrinks += 1
                    budget = max(_MIN_BUDGET, int(budget * _OVERFLOW_SHRINK))
                    payload = self._truncate(text, budget)
                    logger.warning("Server rejected input as too long; shrinking budget to %d and retrying", budget)
                    continue
                attempt += 1
                if attempt >= self._max_retries:
                    logger.error("Single encode failed after %d retries: %s", self._max_retries, exc)
                    return None
                wait = 2 ** (attempt - 1)
                logger.warning("Retry %d/%d single: %s (waiting %ds)", attempt, self._max_retries, exc, wait)
                time.sleep(wait)

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
                # A context-overflow 400 means one item is still too long; retrying the whole
                # batch won't help. Go straight to per-item encoding, which shrinks and retries.
                if _is_context_overflow(exc):
                    logger.warning("Batch rejected as too long; encoding items individually with shrink-retry")
                    return [self._encode_single(t) for t in texts]
                if attempt == self._max_retries - 1:
                    logger.warning("Batch failed, falling back to single encoding: %s", exc)
                    return [self._encode_single(t) for t in truncated]
                wait = 2**attempt
                logger.warning("Retry %d/%d after error: %s (waiting %ds)", attempt + 1, self._max_retries, exc, wait)
                time.sleep(wait)
        return [None] * len(texts)
