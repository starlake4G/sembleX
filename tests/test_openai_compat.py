from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from semble.interfaces import EmbeddingFailure


def _fake_response(vectors: list[list[float]]) -> Any:
    response = MagicMock()
    response.data = [
        MagicMock(index=i, embedding=vec) for i, vec in enumerate(vectors)
    ]
    return response


@pytest.fixture
def provider(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Build an OpenAICompatEmbedding with a stub OpenAI client."""
    fake_openai = MagicMock()
    fake_openai.OpenAI.return_value = MagicMock()

    with patch.dict("sys.modules", {"openai": fake_openai}):
        from semble.backends.embedding.openai_compat import OpenAICompatEmbedding

        impl = OpenAICompatEmbedding(
            base_url="http://x",
            api_key="k",
            model="text-embedding-3-small",
            dim=4,
            batch_size=2,
            max_retries=1,
            max_concurrent=1,
            max_context_tokens=128,
        )
    return impl


def test_encode_empty_returns_empty(provider: Any) -> None:
    out = provider.encode([])
    assert out.shape == (0, 4)


def test_encode_single_batch_success(provider: Any) -> None:
    provider._client.embeddings.create.return_value = _fake_response([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ])
    out = provider.encode(["a", "b"])
    assert out.shape == (2, 4)
    # normalize: each row should be unit length
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), [1.0, 1.0], atol=1e-6)


def test_encode_full_failure_raises(provider: Any) -> None:
    provider._client.embeddings.create.side_effect = RuntimeError("boom")
    with pytest.raises(EmbeddingFailure) as info:
        provider.encode(["a", "b"])
    assert info.value.failed_indices == [0, 1]
    assert info.value.partial is None


def test_encode_partial_failure_surfaces(provider: Any) -> None:
    # First batch (2 items) succeeds, second batch (1 item) fails completely on retry.
    calls = {"n": 0}

    def side_effect(*args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 1:
            return _fake_response([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        raise RuntimeError("transient")

    provider._client.embeddings.create.side_effect = side_effect
    with pytest.raises(EmbeddingFailure) as info:
        provider.encode(["a", "b", "c"])
    assert 2 in info.value.failed_indices
    assert info.value.partial is not None
    assert info.value.partial.shape == (2, 4)


def test_context_overflow_shrinks_and_recovers(provider: Any) -> None:
    # Server rejects the input as too long twice, then accepts the shrunken payload —
    # the chunk must be recovered, not dropped.
    calls = {"n": 0}
    err = RuntimeError("Error code: 400 - This model's maximum context length is 128 tokens. (input_tokens)")

    def side_effect(*args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        if calls["n"] <= 2:
            raise err
        return _fake_response([[1.0, 0.0, 0.0, 0.0]])

    provider._token_budget = 4000  # above _MIN_BUDGET so shrink-retry engages
    provider._client.embeddings.create.side_effect = side_effect
    out = provider._encode_single("x" * 10_000)
    assert out is not None
    assert calls["n"] == 3  # two rejections (each shrinks) then success


def test_is_context_overflow_detection() -> None:
    from semble.backends.embedding.openai_compat import _is_context_overflow

    assert _is_context_overflow(RuntimeError("maximum context length is 32768 tokens"))
    assert _is_context_overflow(RuntimeError("parameter=input_tokens, value=32769"))
    assert not _is_context_overflow(RuntimeError("connection reset"))
