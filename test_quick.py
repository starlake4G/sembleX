import time
import sys

print("0. Imports...")
sys.stdout.flush()

from semble import SembleIndex
from semble.config import SembleConfig, EmbeddingConfig, VectorStoreConfig, RerankerConfig

print("1. Building config...")
sys.stdout.flush()

cfg = SembleConfig(
    embedding=EmbeddingConfig(
        backend="openai_compat",
        openai_base_url="http://100.88.1.2:8000/v1",
        openai_api_key="sk-test",
        openai_model="Qwen/Qwen3-VL-Embedding-2B",
        openai_dim=2048,
        batch_size=64,
        max_concurrent=10,
    ),
    vector_store=VectorStoreConfig(
        backend="faiss",
        use_gpu=False,
        index_type="flat",
        metric="ip",
    ),
    reranker=RerankerConfig(backend="rules"),
    cache={"enabled": False},
)

target = r"C:\sync\Cloud\AI\semble\src\semble"
print(f"2. Indexing {target} ...")
sys.stdout.flush()

t0 = time.perf_counter()
index = SembleIndex.from_path(target, config=cfg)
t1 = time.perf_counter()
print(f"   Done: {index.stats.total_chunks} chunks, {index.stats.indexed_files} files in {t1-t0:.1f}s")
sys.stdout.flush()

for q in ["embedding vector store", "file monitor watch", "cache manager persist", "reranker rules score"]:
    t2 = time.perf_counter()
    results = index.search(q, top_k=3)
    t3 = time.perf_counter()
    print(f"\n--- {q!r} ({(t3-t2)*1000:.0f}ms) ---")
    for r in results:
        first = r.chunk.content.strip().split("\n")[0][:90]
        print(f"  {r.chunk.location} [{r.score:.3f}] {first}")
    sys.stdout.flush()

print("\n=== ALL OK ===")
