
<h2 align="center">
  <img width="30%" alt="sembleX logo" src="https://raw.githubusercontent.com/MinishLab/semble/main/assets/images/semble_logo.png"><br/>
  面向 Agent 的高精度代码搜索 — 增强版<br/>
  <sub>比 grep+read 节省约 98% 的 Token</sub>
</h2>

<div align="center">
  <h2>
    <a href="https://github.com/starlake4G/sembleX"><img src="https://img.shields.io/badge/repo-sembleX-blue" alt="GitHub Repo"></a>
    <a href="https://github.com/starlake4G/sembleX/blob/main/LICENSE">
      <img src="https://img.shields.io/badge/license-MIT-green" alt="License - MIT">
    </a>
  </h2>

[快速开始](#快速开始) •
[MCP 服务器](#mcp-服务器) •
[Bash / AGENTS.md](#bash--agentsmd) •
[命令行工具](#命令行工具) •
[远程服务器](#远程服务器) •
[性能基准](#性能基准) •
[English](README.md)

</div>

SembleX 是 [Semble](https://github.com/MinishLab/semble) 的增强分支，一个专为 AI Agent 构建的代码搜索库。它能即时返回精确的代码片段，比 grep+read 节省约 98% 的 Token。对整个代码库进行端到端的索引和搜索不到一秒即可完成，索引速度比代码专用 Transformer 快约 200 倍，查询速度快约 10 倍，检索质量达到其 99%（见[性能基准](#性能基准)）。所有计算都在 CPU 上运行，无需 API Key、GPU 或外部服务。可以作为 [MCP 服务器](#mcp-服务器)、[独立 HTTP 服务](#远程服务器)运行，或通过 [AGENTS.md](#bash--agentsmd) 在命令行调用，让任何 Agent（Claude Code、Cursor、Codex、OpenCode 等）都能即时访问任意代码库。

## 快速开始

你的 Agent 用自然语言查询 SembleX（例如 `"认证是如何处理的？"`），即可获取相关代码片段，无需 grep 或阅读完整文件。可以通过 MCP 服务器或 AGENTS.md 两种方式接入：

### MCP (Claude Code)

将 SembleX 添加到 Claude Code（需要安装 [uv](https://docs.astral.sh/uv/getting-started/installation/)）：

```bash
claude mcp add semblex -s user -- uvx --from "semblex[mcp]" semblex
```

使用其他 Agent 平台？请参阅下方 [MCP 服务器](#mcp-服务器) 中各平台的配置说明。

### Bash / AGENTS.md

安装 SembleX，然后将以下内容添加到你的 `AGENTS.md` 或 `CLAUDE.md`：

```bash
pip install semblex       # 使用 pip 安装
uv tool install semblex   # 或使用 uv 安装
```

<details>
<summary>AGENTS.md / CLAUDE.md 配置片段</summary>

```markdown
## 代码搜索

使用 `semblex search` 通过描述功能或指定符号/标识符来查找代码，替代 grep：

​```bash
semblex search "认证流程" ./my-project
semblex search "save_pretrained" ./my-project
semblex search "保存模型到磁盘" ./my-project --top-k 10
​```

使用 `semblex find-related` 发现与已知位置相似的代码（传入之前搜索结果中的 `file_path` 和 `line`）：

​```bash
semblex find-related src/auth.py 42 ./my-project
​```

省略 `path` 时默认使用当前目录；也支持 git URL。

如果 `semblex` 不在 `$PATH` 中，请使用 `uvx --from "semblex[mcp]" semblex` 替代。

### 工作流

1. 首先使用 `semblex search` 查找相关代码片段。
2. 仅当返回的代码片段上下文不足时，才查看完整文件。
3. 可选地使用 `semblex find-related`，传入有价值的搜索结果的 `file_path` 和 `line`，发现相关实现。
4. 仅在需要穷举精确匹配或快速确认某个字符串时，才使用 grep。
```

</details>

注意：子 Agent 无法直接调用 MCP 工具，请参阅 [Bash / AGENTS.md](#bash--agentsmd) 和下方的[子 Agent 配置](#子-agent-配置)。

<details>
<summary>更新 SembleX</summary>

```bash
pip install --upgrade semblex   # 使用 pip
uv tool upgrade semblex         # 使用 uv
uv cache clean semblex          # MCP 用户（清理缓存后需重启 MCP 客户端）
```

</details>

## 主要特性

- **快速**：平均仓库索引耗时约 250ms，查询耗时约 1.5ms，全部在 CPU 上运行。
- **精准**：在我们的[性能基准](#性能基准)中 NDCG@10 达到 0.854，与代码专用 Transformer 模型相当，但体积和成本仅为其一小部分。
- **节省 Token**：仅返回相关代码片段，比 grep+read [节省约 98% 的 Token](#性能基准)。
- **零配置**：在 CPU 上运行，无需 API Key、GPU 或外部服务。
- **MCP 服务器**：支持 Claude Code、Cursor、Codex、OpenCode、VS Code 及其他 MCP 兼容的 Agent。
- **本地与远程**：支持本地路径或 git URL。
- **可插拔后端**：通过单个配置文件切换 Embedding（Model2Vec / OpenAI 兼容）、向量存储（NumPy / FAISS / Milvus）和重排序（规则 / 交叉编码器 / 混合）后端。
- **远程服务器模式**：将 SembleX 作为独立 FastAPI 服务运行，使用 Milvus 存储后端、API Key 认证，并支持远程客户端的 CLI/MCP 调用。
- **可配置**：YAML 配置文件，支持环境变量替换。

## MCP 服务器

SembleX 可以作为 MCP 服务器运行，让 Agent 直接搜索任意代码库。仓库按需克隆和索引，索引在会话生命周期内缓存。本地路径会监听文件变更并自动重新索引。

### 配置

> 需要安装 [uv](https://docs.astral.sh/uv/getting-started/installation/)。

<details>
<summary>Claude Code</summary>

```bash
claude mcp add semblex -s user -- uvx --from "semblex[mcp]" semblex
```

</details>

<details>
<summary>Cursor</summary>

添加到 `~/.cursor/mcp.json`（或项目中的 `.cursor/mcp.json`）：

```json
{
  "mcpServers": {
    "semblex": {
      "command": "uvx",
      "args": ["--from", "semblex[mcp]", "semblex"]
    }
  }
}
```

</details>

<details>
<summary>Codex</summary>

添加到 `~/.codex/config.toml`：

```toml
[mcp_servers.semblex]
command = "uvx"
args = ["--from", "semblex[mcp]", "semblex"]
```

</details>

<details>
<summary>OpenCode</summary>

添加到 `~/.opencode/config.json`：

```json
{
  "mcp": {
    "semblex": {
      "type": "local",
      "command": ["uvx", "--from", "semblex[mcp]", "semblex"]
    }
  }
}
```

</details>

<details>
<summary>VS Code</summary>

添加到项目中的 `.vscode/mcp.json`（或用户配置的 `mcp.json`）：

```json
{
  "servers": {
    "semblex": {
      "command": "uvx",
      "args": ["--from", "semblex[mcp]", "semblex"]
    }
  }
}
```

</details>

<details>
<summary>GitHub Copilot CLI</summary>

添加到 `~/.copilot/mcp-config.json`：

```json
{
  "mcpServers": {
    "semblex": {
      "command": "uvx",
      "args": ["--from", "semblex[mcp]", "semblex"]
    }
  }
}
```

</details>

<details>
<summary>Windsurf</summary>

添加到 `~/.codeium/windsurf/mcp_config.json`：

```json
{
  "mcpServers": {
    "semblex": {
      "command": "uvx",
      "args": ["--from", "semblex[mcp]", "semblex"]
    }
  }
}
```

</details>

<details>
<summary>Gemini CLI</summary>

添加到 `~/.gemini/settings.json`：

```json
{
  "mcpServers": {
    "semblex": {
      "command": "uvx",
      "args": ["--from", "semblex[mcp]", "semblex"]
    }
  }
}
```

</details>

<details>
<summary>Kiro</summary>

添加到 `~/.kiro/settings/mcp.json`（或项目中的 `.kiro/settings/mcp.json`）：

```json
{
  "mcpServers": {
    "semblex": {
      "command": "uvx",
      "args": ["--from", "semblex[mcp]", "semblex"]
    }
  }
}
```

</details>

<details>
<summary>Zed</summary>

添加到 `~/.config/zed/settings.json`（或项目中的 `.zed/settings.json`）：

```json
{
  "context_servers": {
    "semblex": {
      "command": "uvx",
      "args": ["--from", "semblex[mcp]", "semblex"]
    }
  }
}
```

</details>


### 工具列表

| 工具 | 说明 |
|------|------|
| `search` | 使用自然语言或代码查询搜索代码库。`repo` 参数可以是本地目录路径或 https:// git URL。 |
| `find_related` | 给定文件路径和行号，返回与该位置代码语义相似的代码片段。 |


<a id="bash-agentsmd"></a>

## Bash / AGENTS.md

MCP 的替代方案是通过 Bash 调用 SembleX。子 Agent 无法直接调用 MCP 工具，因此这是子 Agent 的唯一选择；也可以与顶层 Agent 的 MCP 配合使用。

要添加 Bash 支持，请将以下内容追加到你的 `AGENTS.md`、`CLAUDE.md`、`GEMINI.md` 或类似文件：

```markdown
## 代码搜索

使用 `semblex search` 通过描述功能或指定符号/标识符来查找代码，替代 grep：

​```bash
semblex search "认证流程" ./my-project
semblex search "save_pretrained" ./my-project
semblex search "保存模型到磁盘" ./my-project --top-k 10
​```

使用 `semblex find-related` 发现与已知位置相似的代码（传入之前搜索结果中的 `file_path` 和 `line`）：

​```bash
semblex find-related src/auth.py 42 ./my-project
​```

省略 `path` 时默认使用当前目录；也支持 git URL。

如果 `semblex` 不在 `$PATH` 中，请使用 `uvx --from "semblex[mcp]" semblex` 替代。

## 工作流

1. 首先使用 `semblex search` 查找相关代码片段。
2. 仅当返回的代码片段上下文不足时，才查看完整文件。
3. 可选地使用 `semblex find-related`，传入有价值的搜索结果的 `file_path` 和 `line`，发现相关实现。
4. 仅在需要穷举精确匹配或快速确认某个字符串时，才使用 grep。
```

### 子 Agent 配置

Claude Code、Gemini CLI、Cursor、OpenCode、GitHub Copilot CLI 和 Kiro 都支持专用的 semblex 搜索子 Agent。在项目根目录运行一次 `semblex init`：

```bash
semblex init                      # Claude Code  → .claude/agents/semblex-search.md
semblex init --agent gemini       # Gemini CLI   → .gemini/agents/semblex-search.md
semblex init --agent cursor       # Cursor       → .cursor/agents/semblex-search.md
semblex init --agent opencode     # OpenCode     → .opencode/agents/semblex-search.md
semblex init --agent copilot      # Copilot CLI  → .github/agents/semblex-search.md
semblex init --agent kiro         # Kiro         → .kiro/agents/semblex-search.md
```

如果 semblex 不在 `$PATH` 中，请在命令前加 `uvx --from "semblex[mcp]"`。

## 命令行工具

SembleX 同时提供独立的命令行工具，适用于脚本场景或不需要 MCP 会话的搜索需求。

```bash
# 搜索本地仓库
semblex search "认证流程" ./my-project

# 搜索符号或标识符
semblex search "save_pretrained" ./my-project

# 搜索远程仓库（按需克隆）
semblex search "保存模型到磁盘" https://github.com/MinishLab/model2vec

# 限制返回数量
semblex search "保存模型到磁盘" ./my-project --top-k 10

# 查找与已知位置相似的代码
semblex find-related src/auth.py 42 ./my-project
```

省略 `path` 时默认使用当前目录；也支持 git URL。如果 `semblex` 不在 `$PATH` 中，请使用 `uvx --from "semblex[mcp]" semblex` 替代。

<details>
<summary>Token 节省统计</summary>

`semblex savings` 显示 semblex 在所有搜索中节省的 Token 数量：

```bash
semblex savings           # 按时间段汇总
semblex savings --verbose # 同时显示按调用类型的明细
```

```
  SembleX Token 节省统计
  ════════════════════════════════════════════════════════════════
  时段          调用数   节省量
  ────────────────────────────────────────────────────────────────
  今日          42      [███████████████░]  ~58.4k tokens (95%)
  近 7 天       287     [██████████████░░]  ~312.4k tokens (90%)
  全部          1.4k    [██████████████░░]  ~1.2M tokens (89%)
```

节省量的计算方式：对于每次调用，semblex 记录包含返回代码片段的唯一文件的总字符数和返回片段的字符数。估算节省的 Token 数为 `(文件字符数 − 片段字符数) / 4`（每 4 个字符约等于 1 个 Token）。这是一个保守估算；基线是完整阅读匹配文件，这也是编程 Agent 探索不熟悉代码时的常见做法。

统计数据存储在 `~/.semblex/savings.jsonl`。

</details>

<details>
<summary>Python 库用法</summary>

SembleX 也可以作为 Python 库使用，适用于构建自定义工具或将搜索直接集成到你的代码中。

```python
from semble import SembleIndex

# 索引本地目录
index = SembleIndex.from_path("./my-project")

# 索引远程 git 仓库
index = SembleIndex.from_git("https://github.com/MinishLab/model2vec")

# 使用自然语言或代码查询进行搜索
results = index.search("保存模型到磁盘", top_k=3)

# 查找与特定结果相似的代码
related = index.find_related(results[0], top_k=3)

# 每个结果提供匹配的代码片段
result = results[0]
result.chunk.file_path   # "model2vec/model.py"
result.chunk.start_line  # 127
result.chunk.end_line    # 150
result.chunk.content     # "def save_pretrained(self, path: PathLike, ..."
```

</details>

<a id="远程服务器"></a>

## 远程服务器

SembleX 可以作为独立的 FastAPI HTTP 服务运行，支持通过网络进行远程索引和搜索。适用于团队共享场景或希望将索引工作卸载到专用机器的情况。

### 快速启动

```bash
# 1. 启动 Milvus（向量数据库）
docker run -d --name milvus -p 19530:19530 milvusdb/milvus:standalone

# 2. 启动 SembleX 服务器
semblex server --config examples/openai_compat.yaml

# 3. 索引仓库
semblex index https://github.com/some-org/some-repo

# 4. 搜索
semblex search "认证流程" https://github.com/some-org/some-repo
```

### 配置文件

创建 YAML 配置文件（参考 `examples/openai_compat.yaml`）：

```yaml
index:
  backend: remote            # "local" 直接索引，"remote" 使用服务器

embedding:
  backend: openai_compat     # 或 "model2vec"
  openai_base_url: https://api.openai.com/v1
  openai_api_key: ${SEMBLE_EMBED_API_KEY:-}

vector_store:
  backend: faiss             # 或 "numpy"

reranker:
  backend: rules             # 或 "cross_encoder" / "hybrid"

remote:
  base_url: http://127.0.0.1:8080
  api_key: ${SEMBLE_REMOTE_API_KEY:-}

server:
  host: 0.0.0.0
  port: 8080
  api_key: ${SEMBLE_SERVER_API_KEY:-}

milvus:
  uri: http://127.0.0.1:19530
  collection: semblex_chunks
```

### API 接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/v1/index` | POST | 索引仓库 |
| `/v1/search` | POST | 搜索已索引的仓库 |
| `/v1/find-related` | POST | 查找相关代码片段 |

配置了 `server.api_key` 后，所有接口均支持 Bearer Token 认证。

## 性能基准

我们在 19 种语言的 63 个仓库上使用约 1,250 条查询进行了质量和速度基准测试（左），以及与 grep+read 在相同召回率下的 Token 效率对比（右）。

<table>
<tr>
<td><img src="https://raw.githubusercontent.com/MinishLab/semble/main/assets/images/speed_vs_ndcg_cold.png" alt="速度 vs 质量"></td>
<td><img src="https://raw.githubusercontent.com/MinishLab/semble/main/assets/images/token_efficiency.png" alt="Token 效率：召回率 vs 检索 Token 数"></td>
</tr>
</table>

质量基准（左）衡量检索质量（NDCG@10）与总延迟的关系；semblex 达到了 1.37 亿参数 [CodeRankEmbed](https://huggingface.co/nomic-ai/CodeRankEmbed) Hybrid 99% 的质量，而索引速度快 218 倍。Token 效率基准（右）衡量每种方法在给定召回率下所需的 Token 数；semblex 平均减少 98% 的 Token，仅需 2k Token 即可达到 94% 的召回率，而 grep+read 需要完整的 100k 上下文窗口才能达到 85%。详见 [benchmarks/README.md](benchmarks/README.md) 获取各语言结果、消融实验和完整方法。

## 工作原理

SembleX 使用 [tree-sitter](https://github.com/tree-sitter/py-tree-sitter) 将每个文件拆分为代码感知的代码片段，然后使用两个互补的检索器对每个查询进行评分：使用代码专用 [potion-code-16M](https://huggingface.co/minishlab/potion-code-16M) 模型的静态 [Model2Vec](https://github.com/MinishLab/model2vec) 嵌入用于语义相似性，以及 [BM25](https://github.com/xhluca/bm25s) 用于标识符和 API 名称的词法匹配。两组分数通过 Reciprocal Rank Fusion (RRF) 进行融合。

融合后，结果会通过一组代码感知的信号重新排序：

<details>
<summary><b>排序信号</b></summary>

- **自适应权重。** 符号类查询（`Foo::bar`、`_private`、`getUserById`）获得更多词法权重，而自然语言查询在语义和词法检索器之间保持平衡。
- **定义提升。** 定义了查询符号的代码片段（`class`、`def`、`func` 等）排名高于仅引用它的代码片段。
- **标识符词干。** 查询词经过词干提取后与代码片段中的标识符词干匹配，为包含它们的代码片段提供额外权重。例如，查询 `parse config` 会提升包含 `parseConfig`、`ConfigParser` 或 `config_parser` 的代码片段。
- **文件一致性。** 当同一文件的多个代码片段与查询匹配时，该文件会被提升，使顶部结果反映广泛的文件级相关性，而非单个脱离上下文的代码片段。
- **噪声惩罚。** 测试文件、`compat/`/`legacy/` 兼容层、示例代码和 `.d.ts` 声明存根会被降权，以确保规范实现优先显示。

</details>

由于嵌入模型是静态的，查询时无需 Transformer 前向传播，所有操作都在 CPU 上以毫秒级速度完成。

## 本地模式 vs 远程模式

| 能力 | 本地模式 | 远程模式 |
|------|---------|---------|
| Dense 向量存储 | NumPy / FAISS | Milvus |
| BM25（稀疏） | 磁盘索引 | 磁盘索引（按仓库隔离） |
| 重排序（rules / cross / hybrid） | 支持 | 支持 |
| 文件监控（watchfiles） | 支持 | 不支持 |
| git URL 缓存失效 | 基于 hash | 基于 commit_sha |
| 多仓库隔离 | 单命名空间 | 按 repo_id 隔离 |

## 许可证

MIT

## 引用

如果你在研究中使用了 SembleX，请引用原始 Semble：

```bibtex
@software{minishlab2026semble,
  author       = {{van Dongen}, Thomas and Stephan Tulkens},
  title        = {Semble: Fast and Accurate Code Search for Agents},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.19785932},
  url          = {https://github.com/MinishLab/semble},
  license      = {MIT}
}
```
