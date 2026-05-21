# Semble Refactor — Follow-up TODO

延续 `C:\Users\zz\.claude\plans\serene-yawning-zebra.md` 中的重构方案。Phase 1–6 已落地（详见 `git log` 与该 plan 文件）；以下是 Plan 中标注但**尚未实现**的项，建议按顺序在独立 PR 中补完。

---

## 1. Milvus VectorStore 单测 — `tests/test_milvus_store.py`

**目标**：用 `unittest.mock` 替换 `pymilvus.MilvusClient` + `DataType`，验证 `MilvusVectorStore` 的接口契约，不依赖真实 Milvus。

**覆盖点**：
- `__init__` → `_ensure_collection`
  - 已存在 collection 时只调 `load_collection`，不重建 schema。
  - 不存在时调用 `create_schema` → 添加 `chunk_id` (PK) / `namespace` / 向量字段 → `create_collection(consistency_level=...)`。
- `add(namespace, chunk_ids, vectors)`
  - rows 数量等于 `len(chunk_ids)`；每行包含 `chunk_id`, `namespace`, `<vector_field>`。
  - `chunk_ids` 与 `vectors` 长度不一致时抛 `ValueError`。
  - `metric_type == "IP"` 时手动归一化（断言传给 client 的向量是单位长度）；`COSINE` 时不归一化。
- `delete(namespace, chunk_ids)` → `client.delete(filter=...)` 拼装：`namespace == "..."` && `chunk_id in ["a","b"]`。
- `clear(namespace)` → `client.delete(filter='namespace == "..."')`。
- `query(namespace, vector, k, selector_ids=None)`
  - 单 namespace：filter 仅含 namespace；返回 `[(chunk_id, similarity), ...]`。
  - 带 `selector_ids`：filter 拼上 `chunk_id in [...]`；空 `selector_ids` → 返回 `[]`。
  - L2 度量：相似度走 `1/(1+distance)` 路径；IP/COSINE：直接透传 distance。
  - 处理 pymilvus 两种返回结构：`hit` 为 dict 与对象（`getattr` 路径）。

**关键参考**：
- `src/semble/backends/vector_store/milvus_store.py:13`
- mock 模板：`tests/test_remote_client.py` 里 `patch.dict("sys.modules", ...)` 的方式可借鉴；Milvus 的 mock 可用：
  ```python
  fake_pymilvus = MagicMock()
  with patch.dict("sys.modules", {"pymilvus": fake_pymilvus}):
      from semble.backends.vector_store.milvus_store import MilvusVectorStore
      store = MilvusVectorStore(MilvusConfig(...), dim=4)
      # store._client == fake_pymilvus.MilvusClient.return_value
  ```

---

## 2. Server Indexer 单测 — `tests/test_server_indexer.py`

**目标**：在 tmp work_dir + fake MilvusClient 下验证 `RemoteIndexer` 的全链路（无真实 Milvus / 无真实 embedding）。

**Fixture 思路**：
1. `mock_model` 复用 `tests/conftest.py:81`（256 维确定性向量）。
2. monkeypatch `create_embedding_provider` 让 server 拿到 `mock_model`。
3. monkeypatch `pymilvus` 让 `MilvusVectorStore` 拿到内存版假 client（可以用 `MagicMock` 记录 `insert/delete/search` 调用，或重新实现一个本地 dict）。
4. `tmp_path` 当 `server.work_dir`，构造 `SembleConfig(server={"work_dir": tmp_path})`。

**用例**：
- `index_repo` 幂等：第二次调用返回 `indexed=False`，`chunk_count` 不变。
- `index_repo(force=True)`：清旧再建；validate `MilvusClient.delete` 用 `namespace == repo_id` filter 被调用。
- 空目录抛 `ValueError`（`No supported files found under ...`）。
- 切换 embedding 模型 → 新 `repo_id`，旧 `repo_id` 的 sparse 目录（`work_dir/repos_index/<old>/`）被 `shutil.rmtree`，SQLite `repos` 表里旧条目被清除。
- `search` 未索引时抛 `RepoNotIndexedError(repo_id=..., source=...)`。
- `search` 已索引时跑 hybrid + rules reranker，返回 `list[SearchResult]`，所有 chunk 来自这个 repo。
- `find_related(file_path, line)`：
  - 行号命中 → 返回不包含 target 自身的相关 chunks。
  - 行号未命中 → 返回 `[]`。
- `_clone_or_update`：
  - 通过 monkeypatch `subprocess.run` 模拟 git。
  - 首次：`git clone --depth 1 [--branch ref] -- url repo_dir`。
  - 已存在：先 `git fetch --depth 1 origin <ref or HEAD>` 再 `git reset --hard FETCH_HEAD`；fetch 失败时降级到 stale clone（不抛错）。
  - `--force`：先 `rmtree` 再 clone。
- `_git_head_sha`：成功路径写入 `commit_sha`，可在 `MetadataStore.repos` 行里读到。

**关键参考**：
- `src/semble/server/indexer.py:53`（`RemoteIndexer`）
- `src/semble/server/metadata.py:21`（`MetadataStore`）
- `src/semble/server/metadata.py:6`（`RepoNotIndexedError`）

---

## 3. Server App 集成测 — `tests/test_server_app.py`

**目标**：用 FastAPI 的 `TestClient` 验证 HTTP 层。

**Fixture 思路**：构造 `RemoteIndexer` 用上一项的 fake；用 `create_app(cfg)` + `fastapi.testclient.TestClient`。

**用例**：
- `GET /health` → 200, `{"status":"ok"}`。
- 未配置 `server.api_key` 时所有路由不需要 bearer。
- 配置了 api_key：
  - 无 `Authorization` 头 → 401。
  - 错误 token → 401（断言 `compare_digest` 路径不发生 timing 差异 — 至少行为正确）。
  - 正确 `Bearer <key>` → 200。
- `POST /v1/index` 透传 `ref` 字段到 `RemoteIndexer.index_repo(ref=...)`。
- `POST /v1/search` 未索引 → 409，body `{"error":"not_indexed", ...}`。
- `POST /v1/search` 已索引 → 200，返回 `{"results": [{"score":..., "chunk":{...}}]}`。
- `POST /v1/find-related` 找不到 chunk → 200, `{"results": []}`。
- 已删除的 `POST /v1/find_related` 别名 → 404/405（防回归）。
- `FileNotFoundError` / `NotADirectoryError` / `ValueError` → 400。
- 其它 `Exception` → 500。

**关键参考**：`src/semble/server/app.py:13`。

---

## 4. Monitor 增量更新单测扩展 — `tests/test_monitor.py`（新文件）

**目标**：覆盖 `FileMonitor._on_change` 在新 chunk_id 抽象下的正确性，确保旧的 keep_mask 位置漂移 bug 不复现。

**用例**：
- **add**：tmp_path 起初有 `a.py`，建索引；新增 `b.py` 后触发 `_on_change`；`index.chunks` 包含两个文件的 chunk，`index._semantic_index` 用新 chunk_id 命中 b 的 chunk。
- **remove**：删除 `b.py` → 索引中不再包含其 chunk_id；`vector_store.query` 不再返回。
- **modify**：改 `a.py` 内容 → 旧 chunk_id 消失，新 chunk_id 出现。
- **embedding 部分失败**：mock_model 抛 `EmbeddingFailure(partial=..., failed_indices=[1])` → 成功 chunk 入索引，失败 chunk 跳过，日志记录 dropped 数。
- **embedding 完全失败**：抛 `EmbeddingFailure(partial=None)` → 不抛出，老索引仍然可查（不污染）。

**关键参考**：`src/semble/monitor.py:25`。

---

## 5. CLI Remote 子命令端到端测 — 扩展 `tests/test_cli.py`

**用例**：
- `semble index <repo> --ref main` 透传到 `RemoteSembleClient.index(ref="main")`。
- `cfg.index.backend == "remote"` 时 `search` / `find-related` 走 `_run_remote_cli`；本地不会再尝试 `SembleIndex.from_path`。
- 远程 409 → CLI 打到 stderr 的提示包含 "not indexed"，exit code 1。

---

## 6. 文档

- **`README.md`**：新一节 "Local vs Remote mode"。表格列：
  | 能力 | Local | Remote |
  |---|---|---|
  | Dense | ✅ Numpy / FAISS | ✅ Milvus |
  | BM25 (sparse) | ✅ | ✅ (per-repo on disk) |
  | Reranker (rules / cross / hybrid) | ✅ | ✅ |
  | 文件监控 (watchfiles) | ✅ | ❌（只对 server 本地 path 有意义） |
  | git URL 缓存失效 | hash-based | commit_sha-based |
  | 多租户隔离 | 单 namespace | 按 repo_id |
- **`examples/openai_compat.yaml`** 已存在，README 可指向。
- 列出 server 启动流程：`docker run milvusdb/milvus:standalone` → `semble server --config examples/openai_compat.yaml` → `semble index <repo>` → `semble search ...`。

---

## 7. 已知 ruff 噪音（非阻塞）

- `D102` / `D103` / `D107`：缺失 docstring，主要在 backends/ 与 tests/。重构前就有，建议要么放进 ruff `per-file-ignores` 里要么集中补一波。
- `C901`：`create_app`（15）/ `monitor._on_change`（13）/ `_cli_main`（11）复杂度超阈值，可考虑拆函数（先 monitor，影响最小）。

---

## 8. 验证流程（每个 PR 完成后跑）

```bash
PYTHONPATH=src /c/Users/zz/miniconda3/python -m pytest tests/ -q
PYTHONPATH=src /c/Users/zz/miniconda3/python -m ruff check --select F,E,I src/ tests/
```

端到端（需要 Docker + Milvus）：
```bash
docker run -d --name milvus -p 19530:19530 milvusdb/milvus:standalone
SEMBLE_CONFIG=examples/openai_compat.yaml semble server &
semble index .
semble search "RemoteIndexer hybrid" .
curl -sf -X POST http://127.0.0.1:8080/v1/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"x","repo":"/nope"}'   # 期望 409
```

---

## 9. 死代码清理 / 公开 API 完整性

**死代码**（重构后无 src 引用，但文件还在）：
- 删除 `src/semble/index/dense.py`
  - 含 `SelectableBasicBackend` / `load_model` / `embed_chunks` / `_DEFAULT_MODEL_NAME`
  - 把 `_DEFAULT_MODEL_NAME = "minishlab/potion-code-16M"` 迁到 `EmbeddingConfig` 字段默认值附近（其实它已经是 `EmbeddingConfig.model` 的默认值），或新增 `semble.constants` 模块导出
  - 更新 4 个 benchmarks 的 import：
    - `benchmarks/baselines/ablations.py:21`
    - `benchmarks/token_efficiency.py:27`
    - `benchmarks/speed_benchmark.py:14`
    - `benchmarks/run_benchmark.py:20`
  - 覆盖率报告里 `dense.py 47 lines 0%` 就是这事的暴露
- 删除 `src/semble/index/sparse.py:selector_to_mask`
  - 旧 BM25 `weight_mask` 路径专用，新 `Bm25sSparseIndex.query` 自己处理 selector
  - 保留 `BM25TokenCorpus` / `enrich_for_bm25` / `tokenize_for_bm25`（仍被 sparse 后端使用）

**公开 API 完整性**：
- `src/semble/__init__.py` 增加导出：`SparseIndex`, `LOCAL_NAMESPACE`, `EmbeddingFailure`, `chunk_id`
  - 当前 `__all__` 仅含 `Chunk / EmbeddingProvider / IndexStats / Reranker / SearchResult / SembleIndex / VectorStore`
  - 外部要实现自定义 sparse backend 现在没法 `from semble import SparseIndex`
- `src/semble/backends/__init__.py` 增加导出：`create_sparse_index`
  - 当前只导 `create_embedding_provider / create_vector_store / create_reranker`，缺 sparse
- `SembleConfig` 各子模型加 `model_config = ConfigDict(extra='forbid')`
  - Phase 1 删了 `ServerConfig.max_file_bytes` / `embed_batch_size`
  - Pydantic v2 默认 `extra='ignore'`，旧 yaml 仍含这两个字段会**静默丢弃**
  - 改成 `forbid` 让旧配置直接 ValidationError，让用户被迫迁移到 `indexing.*`

---

## 10. 行为对齐 / 一致性

**远程 vs 本地 parity 漏洞**：
- `RemoteIndexer.find_related` 加 language selector
  - 本地 `SembleIndex.find_related` 用 `target.language` 构造 `selector_ids`（见 `src/semble/index/index.py:193`）
  - 远程版当前直接 `self._vector_store.query(repo_id, query_vector, k=top_k + 1)`，无 selector（见 `src/semble/server/indexer.py:150`）
  - 实现：`MetadataStore` 新增 `chunk_ids_in_language(repo_id: str, language: str) -> list[str]`
    - SQL：`select chunk_id from chunks where repo_id = ? and language = ?`
    - 已有 `idx_chunks_repo` 但没语言索引，可以加 `create index if not exists idx_chunks_lang on chunks(repo_id, language)`
  - 在 `RemoteIndexer.find_related` 里：`selector_ids = self._metadata.chunk_ids_in_language(repo_id, target.language) if target.language else None`，传给 `_vector_store.query`

**Milvus 向量归一化**：
- `MilvusVectorStore._prepare_vectors` 当前只对 `metric_type == "IP"` 归一化
  - COSINE 度量下 pymilvus 查询时归一化，但**训练阶段**（IVF/HNSW）使用未归一化向量会让聚类中心分布偏离单位球面，影响近似召回精度
  - 改为：`if self._config.metric_type in ("IP", "COSINE"):` 都归一化
  - L2 不动

**Cache 失效**：
- `cache/manager.py:is_disk_valid` 的 git URL 分支二选一：
  - 选 A：删掉 `if root is None: return True`，让 git URL 永远走 hash 路径（实际未命中 → 重建）
  - 选 B：按 Plan 写 `commit_sha` 到 `meta.json`，加载时比对当前 git URL 对应的 commit_sha
  - 由于 `SembleIndex.from_git` 用 `TemporaryDirectory` 从不写盘，**当前这段是死代码**，**选 A 更诚实**

**Sparse 边界**：
- `Bm25sSparseIndex.save` 允许空索引
  - 现状：`if self._index is None: raise RuntimeError("Cannot save an empty Bm25sSparseIndex")`
  - 调用方风险：`monitor._on_change` 在 chunks 全被删完后会 `_sparse_index.build([], [])` → 接着 `save_to_disk` → save 抛错
  - 解法：`build` 阶段对空集做合法处理（写一个 `.empty` marker），`save/load` 兼容空状态

**Server 鲁棒性**：
- `RemoteIndexer._open` 当 SQLite 有 repo_id 但 sparse 文件缺失/损坏时崩
  - 现在直接 `Bm25sSparseIndex().load(self._sparse_path(repo_id))` 会 raise FileNotFoundError
  - 包成 `RepoNotIndexedError(repo_id, source=...)` 抛出，让 `app.py` 翻译成 409 + 让 client 自动提示用户重跑 index

---

## 11. 鲁棒性 / 可观察性

**内存上限**：
- `_LoadedIndex` 把整个 repo 的 `chunks: list[Chunk]` + `by_id: dict[str, Chunk]` 装入内存
  - 大 repo（10w+ chunks，每个 ~2KB content）会让 4 个 LRU slot 占用 GB 级
  - 加 `RemoteIndexer.max_loaded_chunks: int = 200_000` 阈值；超过则 fail-loud 抛出明确异常，提示 server 端需要更大内存或拆分仓库

**预存在 bug 确认**：
- `tests/test_cli.py:142` `_run_init(agent=agent)` 里 `agent` 未定义
  - 这条记录在 staged diff 里就这样了，不是 Phase 1-6 引入的
  - pytest 全绿可能是该用例在某种 import path 下被跳过，需要单独跑 `pytest tests/test_cli.py::test_init_creates_file -v` 确认
  - 若确认未运行，加 `@pytest.mark.parametrize("agent", list(Agent))` 让它真正跑起来；同时检查相邻 `test_init_refuses_overwrite_without_force` / `test_init_overwrites_with_force` 是否同样有问题

---

## 推荐落地顺序

1. **第 9 节（死代码 + 公开 API）** —— 都是机械改动，半小时
2. **第 10 节里的 Cache 失效 + Sparse 边界** —— 简单 + 防 monitor 路径上的崩
3. **第 10 节里的 Milvus 归一化** —— 一行改动，但要让 1 节里写好的 Milvus 单测覆盖到
4. **第 10 节里的 find_related parity** —— 需要 MetadataStore schema 微调（加 index）
5. **第 11 节的 max_loaded_chunks** —— 等 Milvus 集成测跑通再加，避免过早优化
6. **第 11 节的 test_cli pre-existing bug** —— 单独 PR 修，不阻塞其他
