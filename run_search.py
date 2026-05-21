import sys
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

from semble import SembleIndex
from semble.config import SembleConfig, EmbeddingConfig, VectorStoreConfig, RerankerConfig

cfg = SembleConfig(
    embedding=EmbeddingConfig(
        backend="openai_compat",
        openai_base_url="http://100.88.1.2:8000/v1",
        openai_api_key="sk-test",
        openai_model="Qwen/Qwen3-VL-Embedding-2B",
        openai_dim=2048,
        batch_size=128,
        max_concurrent=20,
    ),
    vector_store=VectorStoreConfig(
        backend="faiss",
        use_gpu=False,
        index_type="flat",
        metric="ip",
    ),
    reranker=RerankerConfig(backend="rules"),
    cache={"enabled": True},
)

target = sys.argv[1] if len(sys.argv) > 1 else r"C:\sync\Cloud\AI"
queries = sys.argv[2:] if len(sys.argv) > 2 else [
    "video analysis object detection YOLO",
    "face recognition identity matching",
    "trading strategy backtesting",
    "code search embedding vector FAISS",
    "MCP server tool agent",
    "pipeline step orchestration",
    "authentication token verification",
    "configuration loading YAML",
]

print(f"=== Indexing {target} ===")
t0 = time.perf_counter()
index = SembleIndex.from_path(target, config=cfg)
t1 = time.perf_counter()
print(f"Done: {index.stats.total_chunks} chunks, {index.stats.indexed_files} files, {len(index.stats.languages)} langs in {t1-t0:.1f}s")
print(f"Languages: {dict(sorted(index.stats.languages.items(), key=lambda x: -x[1])[:10])}")
print()

for q in queries:
    print(f"--- {q!r} ---")
    t2 = time.perf_counter()
    results = index.search(q, top_k=3)
    t3 = time.perf_counter()
    print(f"  {len(results)} results in {(t3-t2)*1000:.0f}ms")
    for r in results:
        loc = r.chunk.location.replace(str(target), ".", 1)
        first = r.chunk.content.strip().split("\n")[0][:100]
        print(f"  {loc} [{r.score:.3f}]  {first}")
    print()
