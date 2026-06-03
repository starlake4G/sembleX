<h2 align="center">
  SembleX — 面向 AI Agent 的跨仓库代码检索<br/>
  <sub>一次索引大量仓库，随后在全部仓库范围内搜索并查找相似实现</sub>
</h2>

<div align="center">

[它能做什么](#它能做什么) •
[架构](#架构) •
[前置条件](#前置条件) •
[安装](#安装) •
[配置](#配置) •
[快速上手](#快速上手) •
[MCP 集成](#mcp-集成) •
[CLI 参考](#cli-参考) •
[工作原理](#工作原理) •
[English](README.md)

</div>

SembleX 是对 [Semble](https://github.com/MinishLab/semble) 的激进改造分支，只为一件事服务：**为 AI Agent 提供一个跨大量代码仓库的统一全局索引。** Agent 由此可以 (1) 用自然语言或符号查询一次性搜索所有仓库，(2) 给定任意一处代码位置，在*其它*仓库中找到相似实现作为参考。

它以 **MCP 为核心**：MCP server 是主接口，进程内常驻一份预热索引；CLI 只是用于索引、调试和脚本化的瘦客户端。稠密向量存放在远程 **Milvus** 集合中，chunk 正文与词法（BM25）索引则放在紧凑的**本地核心数据库**里。这种拆分让存储很小（Milvus 只存 `id + namespace + vector`），索引也快（向量由远程的代码专用 embedding 服务产出）。

## 它能做什么

两项能力，同一套引擎：

- **跨库搜索** —— `search("认证是如何处理的")` 从*任意*已索引仓库返回最相关的代码片段，每条都标注其来源仓库（`repo :: file:line`）。
- **跨库找相似** —— `find_related(repo, file, line)` 对该位置的代码做 embedding，返回*其它*仓库中语义相似的实现（默认排除源仓库），让 Agent 参考同类问题在别处是如何解决的。

两者归约为同一条流水线：*产出查询向量 → 全局检索 → 融合 → 排序*。自然语言查询和代码位置，只是产出该向量的两种方式。

## 架构

```
        ┌──────────── 进程内预热索引（MCP 守护）────────────┐
  MCP ─▶│  GlobalIndex                                       │
(核心)  │   • MetadataStore — SQLite：chunk 正文 + repo_id   │─▶ 远程 Milvus
        │   • 全局 BM25     — 单个 IncrementalSparseIndex     │     （单一集合，
  CLI ─▶│   • rules 排序    — 代码感知的加权/降权             │      namespace = repo_id，
(调试) │                                                     │      只存 id + ns + vector）
        └─────────────────────────────────────────────────────┘─▶ 远程 OpenAI 兼容
                                                                    embedding 服务
```

- **单一全局 Milvus 集合**保存所有仓库的稠密向量，以 `namespace = repo_id` 分区。全局搜索时去掉 namespace 过滤；单仓库搜索与 `find_related` 排除源仓库时则用到它。
- **单个全局 BM25 索引**（词法）覆盖全部 chunk，以全局唯一的 `chunk_id` 为键（哈希含 `repo_id`）。重新索引某仓库时，会增量地移除其旧 chunk id 并加入新的。
- **本地核心数据库**（默认 `~/.semble/core`）包含 `metadata.sqlite3`（chunk 正文 + 仓库元数据）和 `sparse/`（持久化的 BM25 索引）。MCP 进程让它们常驻预热；CLI 调用会在 BM25 索引文件 mtime 变化时自动重载。

## 前置条件

与上游 Semble 不同，SembleX **不是**零配置——它依赖你为其指定的两项服务：

1. **一个 Milvus 实例**（本地或远程）。快速起一个本地的：
   ```bash
   docker run -d --name milvus -p 19530:19530 milvusdb/milvus:standalone
   ```
2. **一个 OpenAI 兼容的 embedding 服务**，最好是自建 / 代码专用模型。任何暴露 `POST /v1/embeddings` 的服务都可用（vLLM、TEI、Ollama、OpenAI 本身……）。embedding **维度可配置**——设为与你的模型一致即可。

## 安装

从构建好的 wheel 安装（推荐用于部署——独立安装，不依赖源码树）：

```bash
# 构建一次
uv build --wheel                # 产出 dist/semblex-<version>-py3-none-any.whl

# 安装到你的环境（conda base、venv 等），带全部 extras
pip install "dist/semblex-<version>-py3-none-any.whl[all]"
```

或直接从源码安装：

```bash
pip install ".[all]"            # 或：uv tool install ".[all]"
```

Extras：`mcp`（MCP server）、`yaml`（YAML 配置）、`tokenizer`（tiktoken 计 token）、`all`（= 以上三者）。命令行入口是 `semble`。

## 配置

创建 `semble.yaml`（YAML 或 JSON；支持 `${VAR:-default}` 形式的环境变量替换）。所有键都有默认值，下面只列你通常需要设置的：

```yaml
embedding:
  backend: openai_compat
  openai_base_url: http://your-embed-host:8000/v1   # OpenAI 兼容服务地址
  openai_api_key: ${SEMBLE_EMBED_KEY:-dummy}        # 许多自建服务会忽略此项
  openai_model: your-code-embedding-model
  openai_dim: 2048                                  # 必须与模型输出维度一致
  max_context_tokens: 32768

milvus:
  uri: http://your-milvus-host:19530
  collection: semble_global
  metric_type: IP                                   # 向量已 L2 归一化 → IP 等价于 cosine

core:
  dir: ~/.semble/core                               # 本地 SQLite + BM25 + git 克隆

reranker:
  backend: rules                                    # 轻量级代码感知排序
```

配置通过 `--config <path>`、`SEMBLE_CONFIG` 环境变量，或自动发现（`./semble.yaml`、`~/.config/semble/config.yaml`、`~/.semble/config.yaml` 等）加载。

## 快速上手

```bash
# 1. 注册仓库。可以逐个添加……
semble --config semble.yaml projects add /path/to/repo-a
semble --config semble.yaml projects add https://github.com/some-org/repo-b
# ……或扫描一个目录树，注册其下所有 git 仓库：
semble --config semble.yaml projects scan /path/to/all/my/repos --depth 3

# 2. 构建全局索引（embedding + 写入 Milvus + 构建 BM25）
semble --config semble.yaml index --all

# 3. 在所有已索引仓库范围内搜索
semble --config semble.yaml search "增量稀疏 BM25 索引的增删" -k 8

# 4. 从某条结果的 repo/file/line，在其它仓库找相似代码
semble --config semble.yaml find-related /path/to/repo-a src/foo/chunking.py 42 -k 6

# 5. 查看索引状态
semble --config semble.yaml status
```

要（重新）索引单个仓库，传它的路径/URL 而非 `--all`：`semble index /path/to/repo-a`（未变化则为空操作；用 `reindex` 或 `--force` 强制重建）。

## MCP 集成

MCP server 是主接口。启动时构建一份常驻预热的 `GlobalIndex`，通过 stdio 提供两个工具。

### 工具

| 工具 | 说明 |
|------|------|
| `search(query, top_k=5, scope="global", repo=None)` | 搜索已索引仓库。`scope="global"`（默认）覆盖所有仓库——最适合在别处找参考实现；`scope="workspace"` 限定到 MCP server 工作目录所在的仓库（即当前项目）。`repo` 可按路径/URL 指定具体仓库，优先级高于 `scope`。每条结果标注来源仓库。 |
| `find_related(repo, file_path, line, top_k=5)` | 给定一处位置（用上一条 `search` 结果中的 `repo`/`file_path`/`line`），返回*其它*仓库中语义相似的代码。源仓库被排除。 |

> **作用域。** `scope="workspace"` 会把 MCP server 的启动目录解析到包含它的那个已索引仓库（子目录也行）。MCP 客户端会在当前项目目录下拉起 server，因此这等价于"只搜当前项目"。若该目录未被索引，工具会明确告知并建议改用 `scope="global"`。

### 注册到 Claude Code

```bash
claude mcp add semblex -s user -- semble --config /abs/path/to/semble.yaml serve
```

<details>
<summary>其它 Agent（Cursor、Codex、VS Code……）</summary>

在各 harness 的 MCP 配置中使用同一条命令。例如 Cursor（`~/.cursor/mcp.json`）：

```json
{
  "mcpServers": {
    "semblex": {
      "command": "semble",
      "args": ["--config", "/abs/path/to/semble.yaml", "serve"]
    }
  }
}
```

Codex（`~/.codex/config.toml`）、VS Code（`.vscode/mcp.json`）、Windsurf、Gemini CLI 等模式相同——把 `command` 设为 `semble`（或其绝对路径），`args` 设为 `["--config", "<path>", "serve"]`。

</details>

> 索引在带外通过 CLI 完成（`semble index ...`）。MCP server 读取 CLI 构建的索引；当磁盘上的索引变化时，会自动重载其 BM25 部分。

## CLI 参考

`semble [--config PATH] <command>`

| 命令 | 用途 |
|------|------|
| `serve` | 通过 stdio 运行跨库 MCP server（不带子命令时的默认行为）。 |
| `index [source] [--all] [--ref REF] [--include-text-files] [--force]` | 将仓库（本地路径或 git URL）索引进全局索引，或用 `--all` 索引所有已注册项目。`--ref` 为 git URL 选择分支/标签；`--include-text-files` 一并索引非代码文本；`--force` 强制重建。 |
| `reindex [source] [--all] ...` | `index --force` 的别名。 |
| `search <query> [--scope global\|workspace] [--repo REPO] [-k N]` | 搜索已索引仓库。`--scope workspace` 限定到当前目录所在的仓库；`--repo` 按路径/URL 指定具体仓库（优先级高于 `--scope`）。 |
| `find-related <repo> <file_path> <line> [-k N] [--include-source-repo]` | 查找与某位置相似的代码。`repo` 是搜索结果 `repo:` 行给出的源仓库路径/URL。默认排除源仓库；`--include-source-repo` 则保留。 |
| `status` | 显示仓库数、chunk 数、embedding 模型/维度、Milvus URI 及已注册来源。 |
| `projects add <source> [--name N] [--ref REF] [--include-text-files]` | 注册本地路径或 git URL。 |
| `projects scan <root> [--depth N]` | 注册 `root` 下找到的所有 git 仓库（默认深度 3）。 |
| `projects list` / `projects remove <source>` | 列出 / 注销项目。 |

## 工作原理

SembleX 用 [tree-sitter](https://github.com/tree-sitter/py-tree-sitter) 将每个文件切成代码感知的 chunk，然后用两种互补信号检索，并以 Reciprocal Rank Fusion（RRF）融合：

- **稠密** —— 来自你的 OpenAI 兼容服务的 embedding，L2 归一化后存入 Milvus（用内积实现 cosine）。
- **词法** —— 覆盖标识符与 API 名称的全局 BM25 索引。

融合后，结果用代码感知信号重排（符号/自然语言自适应加权、定义加权、标识符词干匹配、文件一致性，以及对测试/legacy/示例/桩代码的降权）。在跨库模式下，这些排序信号**按仓库分别计算再合并**，因此文件一致性始终限定在仓库内，不同仓库中同名文件也绝不会相互碰撞。

### 存储占用

拆分存储的设计正是 SembleX 体积小的关键。Milvus **只**存 `chunk_id + namespace + vector`，chunk *正文*放在本地 SQLite。作为参照：索引两个仓库（维度 2048、约 600 个 chunk）产生约 **3.9 MB** 本地核心库（1.1 MB SQLite 正文 + 2.9 MB BM25）和约 **4.9 MB** 的 Milvus 原始向量。调小 `embedding.openai_dim` 可进一步压缩。

## 许可

MIT。SembleX 是 MinishLab 团队 [Semble](https://github.com/MinishLab/semble) 的分支；原始的检索与排序工作归属于他们。
