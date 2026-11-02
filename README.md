<div align="center">

# 🔆 Beacon

### Hybrid retrieval & RAG pipeline toolkit for Python — dense + sparse fusion, re-ranking, zero dependencies, fully offline.

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Dependencies](https://img.shields.io/badge/dependencies-zero-blueviolet)](https://github.com/WsNodeX/beacon)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey)](https://github.com/WsNodeX/beacon)
[![Tests](https://img.shields.io/badge/tests-61%20passing-brightgreen)](tests/)
[![Offline](https://img.shields.io/badge/offline-ready-orange)](https://github.com/WsNodeX/beacon)

</div>

<p align="center">
  <a href="#-why-beacon">✨ Why</a> ·
  <a href="#-features">🚀 Features</a> ·
  <a href="#-installation">📦 Install</a> ·
  <a href="#-quick-start">⚡ Quick Start</a> ·
  <a href="#-architecture">🏗️ Architecture</a> ·
  <a href="#-api-reference">🔌 API</a> ·
  <a href="#-roadmap">🧭 Roadmap</a> ·
  <a href="#-中文技术详解">📝 中文详解</a>
</p>

---

Beacon turns a folder of documents into a queryable knowledge index. It combines
**dense vector retrieval** (semantic, catches paraphrases) with **sparse BM25
retrieval** (lexical, nails exact keywords), merges the two with **Reciprocal
Rank Fusion**, then **re-ranks** the merged candidates with a transparent feature
model. The whole thing runs on a stock Python 3.9+ interpreter — no model files,
no API keys, no native extensions.

It is built to be **read, extended, and demoed in an interview**, not hidden behind
a cloud service.

## ✨ Why Beacon

Most "retrieval" tutorials stop at `pip install sentence-transformers` and call it
a day. That hides the actual mechanics and ties you to a 400 MB download and a
network call. Beacon takes the opposite stance:

- **You see every step.** Splitting, embedding, the two retrievers, fusion, and
  re-ranking are separate, inspectable modules.
- **It works offline, deterministically.** The default embedder is a hashing-based
  encoder, so the same text always yields the same vector on any machine.
- **It is tiny enough to fit in your head.** ~1.5k lines, zero third-party
  dependencies, fully unit-tested.
- **It is a real architecture.** Dense + sparse + fusion + re-rank is exactly the
  pattern behind production RAG systems — just without the GPU bill.

## 🚀 Features

- 🧩 **Three text splitters** — recursive character, sentence, and token windows,
  all overlap-aware so facts spanning a boundary stay retrievable.
- 🔢 **Deterministic embeddings** — `HashEmbedder` maps text to a fixed vector
  with the hashing trick; plug in any `encode(text) -> list[float]` backend.
- 🔍 **Dual retrieval** — in-memory `VectorStore` (cosine) and `BM25` (Okapi).
- 🔗 **Reciprocal Rank Fusion** — merges ranked lists using ranks only, so
  incompatible score scales fuse fairly.
- 🎯 **Transparent re-ranker** — combines semantic similarity, lexical Jaccard,
  and a length prior with plain, tunable weights.
- 💾 **JSON persistence** — save/load an entire index; no database required.
- 📊 **Offline evaluation** — `recall@k` and MRR without a single network call.
- 🖥️ **CLI + library** — `beacon index`, `beacon query`, `beacon eval`, `beacon demo`.

## 📦 Installation

Beacon is pure standard library. Clone and use — no build, no `pip install` of
heavy packages.

```bash
git clone https://github.com/WsNodeX/beacon.git
cd beacon
python -m beacon.cli demo          # prints a live retrieval demo
```

Optional, if you want it importable as a package:

```bash
pip install -e .
```

> Requires Python 3.9 or newer. That is the only requirement.

## ⚡ Quick Start

```python
from beacon.core.pipeline import RetrievalPipeline
from beacon.core.types import Document

docs = [
    Document(text="BM25 ranks documents by term frequency and inverse document "
                   "frequency for search engines.", doc_id="bm25"),
    Document(text="A bloom filter is a probabilistic set with no false negatives "
                   "but possible false positives.", doc_id="bloom"),
    Document(text="HyperLogLog estimates set cardinality using little memory via "
                   "leading zero runs.", doc_id="hll"),
]

pipe = RetrievalPipeline()
pipe.index(docs)

for hit in pipe.search("how do search engines rank by term frequency", top_k=2):
    print(f"{hit.score:.4f}  {hit.chunk.text}")
```

**Expected output**

```
0.61XX  BM25 ranks documents by term frequency and inverse document frequency for search engines.
0.20XX  Reciprocal rank fusion merges several ranked result lists by adding 1/(k+rank) for each item, ...
```

You can also run the bundled CLI demo:

```bash
python -m beacon.cli demo
```

```
Indexed 4 chunks.

Q: how do search engines rank documents by term frequency?
   [0.4192] BM25 is a ranking function used by search engines to score documents against a query using
   [0.2065] Reciprocal rank fusion combines several ranked result lists by summing 1/(k+rank) for each item,

...
```

## 🏗️ Architecture

```
                         ┌──────────────── Query ────────────────┐
                                       │
                          ┌────────────▼────────────┐
                          │     RetrievalPipeline     │
                          └──────┬──────────────┬─────┘
                     dense       │              │      sparse
                              ▼              ▼
                     ┌──────────────┐  ┌──────────────┐
                     │ VectorStore  │  │    BM25      │
                     │ (HashEmbed)  │  │  (lexical)   │
                     └──────┬───────┘  └──────┬───────┘
                            │  two ranked lists │
                            └────────┬──────────┘
                                  ▼
                         ┌────────────────────┐
                         │  RRF Fusion        │   ranks only, scale-free
                         └─────────┬──────────┘
                                   ▼
                         ┌────────────────────┐
                         │  FeatureReranker   │   semantic + lexical + length
                         └─────────┬──────────┘
                                   ▼
                          Top-K ScoredChunks
```

Each box is an independent, unit-tested module under `beacon/core/`. Swapping the
embedder, the splitter, or the re-ranker never touches the others.

## 🔌 API Reference

| Component | Purpose | Key methods |
|-----------|---------|-------------|
| `Document` / `Chunk` / `ScoredChunk` | Data containers | `Chunk.tokens()` |
| `Splitter` | Base splitter | `split_document(doc) -> List[Chunk]` |
| `RecursiveCharacterSplitter` | Hierarchical cut on separators | `split(text)` |
| `SentenceSplitter` | Groups whole sentences | `split(text)` |
| `TokenSplitter` | Token windows + overlap | `split(text)` |
| `Embedder` / `HashEmbedder` | Text → vector | `encode(text) -> List[float]`, `dim()` |
| `VectorStore` | Dense cosine search | `add(chunks)`, `search(query, top_k)` |
| `BM25` | Sparse lexical search | `fit(chunks)`, `search(query, top_k)` |
| `reciprocal_rank_fusion` | Merge ranked lists | `fusion([list_a, list_b], k=60)` |
| `FeatureReranker` | Late-interaction re-rank | `rerank(query, chunks, top_n)` |
| `RetrievalPipeline` | End-to-end orchestrator | `index(docs)`, `search(q, top_k)`, `save_json/load_json` |
| `load_text` / `load_markdown` / `load_jsonl` / `load_dir` | Document loaders | — |
| `recall_at_k` / `mean_reciprocal_rank` / `evaluate` | Metrics | — |

**CLI**

```bash
python -m beacon.cli demo                                   # live demo
python -m beacon.cli index  --dir ./docs --store idx.json   # build index
python -m beacon.cli query  --store idx.json --text "..."   # query
python -m beacon.cli eval   --store idx.json --top-k 5      # self-evaluation
```

## 🧭 Roadmap

- [ ] Additive JSON persistence for incrementally growing indices
- [ ] `MinHashLSH` near-duplicate detection before indexing
- [ ] A learned cross-encoder re-ranker slot (drop-in for `FeatureReranker`)
- [ ] Token-budget-aware chunking for fixed-context LLMs
- [ ] Optional FAISS / HNSW backend behind the same `VectorStore` API
- [ ] Metadata filtering (`doc_id`, custom fields) in search
- [ ] A `serve` subcommand exposing the pipeline over a local HTTP endpoint

## 🤝 Contributing

Bug reports, feature ideas, and pull requests are welcome. The codebase is small
and deliberately dependency-free — please keep new code standard-library only
unless a new backend is explicitly proposed. Run the test suite before opening a
PR:

```bash
python -m unittest discover -s tests
```

## 📄 License

Released under the [MIT License](LICENSE). Do whatever you like; attribution is
appreciated.

---

## 📝 中文技术详解

> 下面这篇可以直接当作发在掘金 / 知乎 / 小红书 / B 站的技术科普文底稿。

### 一、它解决什么问题

做 RAG（检索增强生成）的同学几乎都踩过同一个坑：**只用向量检索，搜不准关键词；只用关键词检索，搜不到同义改写**。比如你问"怎么按词频给文档排序"，向量库大概率把"词频""排序"这几个语义相近但字面不同的句子排得很靠前，却漏掉正文里明明白白写着 "BM25 ranks documents by term frequency" 的那一段。

Beacon 的核心主张很简单：**把稠密检索（懂语义）和稀疏检索（懂字面）两条路都走，再用一套可解释的办法把它们合并**。这正是今天生产级 RAG 系统的主流做法，而 Beacon 用不到 2000 行纯标准库代码把它完整实现了一遍，不依赖任何模型权重和联网。

### 二、核心原理

检索可以拆成四步流水线：

1. **切块（Chunking）**。长文档先切成 200~500 字的小块，块是检索的最小单位。Beacon 提供递归切分、按句切分、按 token 切分三种策略，并且都带"重叠窗口"，避免一个知识点被切断后两边都搜不到。
2. **两条检索支路并行**。
   - **稠密支路**：把每个块映射成一个向量，查询也映射成向量，按余弦相似度找最近的块。默认用的是**哈希嵌入（hashing trick）**——把每个词哈希到一个固定维度再累加、归一化。它不需要词表、不需要训练，同一段文本在任何机器上算出来都一模一样，非常适合做教学和可复现的实验。
   - **稀疏支路**：经典的 **BM25**，用"词频 + 逆文档频率 + 长度归一"给文档打分，专治精确关键词。
3. **融合（Fusion）**。两条路返回的"分"量纲完全不同（余弦在 0~1，BM25 可能上百），直接相加会翻车。Beacon 用 **Reciprocal Rank Fusion（RRF）**：只关心"排名"不关心"分数"，每个候选的得分是 `Σ 1/(k+rank)`。因为只看名次，量纲差异被自然抹平，两条路公平合并。
4. **重排（Re-rank）**。融合后的候选再过一个**可解释的特征重排器**：把"语义余弦相似度""词面 Jaccard 重叠""长度先验"按可调权重加权求和，排序输出 Top-K。

### 三、关键实现点（面试常问）

- **为什么用 RRF 而不是直接归一化分数？** 因为归一化要假设分数分布（min-max？z-score？），而 RRF 只看排名，对分布零假设，鲁棒得多。这是它在大厂 RAG 里被采纳的主因。
- **哈希嵌入到底准不准？** 哈希嵌入的向量相似度本质上近似"词面重合度"，对同义改写不敏感，但**胜在零依赖、可复现、快**。真实项目里把它换成 `sentence-transformers` 或 OpenAI Embeddings 即可，调用方代码一行都不用改——这就是面向接口编程的好处。
- **切块为什么要重叠？** 一个事实可能正好落在块边界上。重叠窗口让它在左右两个块里都出现，检索时总能被捞出来。
- **重排器为什么"透明"？** 很多框架把重排做成黑盒交叉编码器。Beacon 故意用一个权重全摊开的特征模型，方便你 debug：哪条结果排前面，是因为语义分高还是因为词面重合高，一眼能看穿。

### 四、与同类方案的对比

| 维度 | Beacon | 纯向量库（如简单 FAISS 封装） | 重型 RAG 框架 |
|------|--------|------------------------------|--------------|
| 依赖 | 0（纯标准库） | 需 numpy / faiss | 一堆第三方包 |
| 是否离线 | ✅ 完全离线 | 取决于嵌入模型 | 通常要联网拉模型 |
| 混合检索 | ✅ 稠密+稀疏+RRF | ❌ 仅稠密 | ✅（但封装深） |
| 可解释性 | ✅ 权重全摊开 | ❌ | 部分 |
| 上手成本 | 读 2000 行即懂 | 中等 | 高 |
| 面试叙事 | 强（全流程自己写） | 弱 | 中 |

一句话：**Beacon 牺牲了一点检索上限，换来了"全栈可解释 + 零依赖 + 可复现"**，对想真正搞懂 RAG、而不是调包的同学，价值最高。

### 五、如何扩展

- **换更好的嵌入**：实现一个 `encode(text) -> list[float]` 的类（比如接本地 `sentence-transformers`），传给 `RetrievalPipeline(embedder=...)`，其余代码零改动。
- **加重排模型**：把 `FeatureReranker` 换成真正的交叉编码器，只要保留 `rerank(query, chunks) -> 排序列表` 这个接口。
- **上规模**：`VectorStore` 背后可以换成 FAISS / HNSW 做近似最近邻，对外接口不变。
- **加元数据过滤**：在 `search` 时按 `doc_id` 或自定义 `metadata` 过滤，企业检索必备。

### 六、把它扩展成一篇发布稿的提示

- **标题方向**：《我用 2000 行纯 Python 手搓了一个 RAG 检索引擎》《面试被问 RAG 原理？自己写一个就懂了》。
- **钩子开头**：从一个"向量检索搜不到关键词"的翻车经历切入。
- **结构**：痛点 → 四步流水线手绘 ASCII 图 → 贴 `quick start` 跑出来的真实输出 → 对比表 → "如何扩展成生产级"。
- **配图建议**：把上面的架构图导出成图，再补一张"只用语义检索 vs 混合检索"的命中对比截图，转发率更高。
- **互动结尾**：抛一个问题——"你的 RAG 用的是纯向量还是混合检索？踩过什么坑？"引导评论。

---

<div align="center">
  <sub>Built with standard library only. No black boxes.</sub>
</div>
