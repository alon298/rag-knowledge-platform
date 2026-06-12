# RAG Knowledge Platform · 企业知识库问答平台

> 个人开源复刻 RAGFlow / FastGPT 核心链路的轻量级 RAG 知识库问答平台。

## 核心特性

- **深度文档解析**：支持 TXT / Markdown / CSV，表格按行结构化，保留标题层级
- **智能分块**：重叠窗口 + 标题感知，避免语义被截断
- **混合检索**：向量召回（本地 TF-IDF 加权，可切换 Embedding API）+ 关键词全文匹配重排序
- **引用溯源**：回答附带来源分块、文档名与块定位，可核验
- **多知识库隔离**：按库独立索引与检索，互不串扰
- **评测与回归**：内置 50+ 条评测集，一键跑召回率 / 命中率回归

## 架构

```
ingest(解析/分块) → embed(向量化) → store(索引持久化)
       ↓
retrieve(Top-K 召回 + 关键词重排) → generate(带引用回答)
```

## 快速开始

```bash
# 1. 准备语料（demo 语料见 data/）
python rag_platform.py ingest data/ --kb demo

# 2. 提问（无 API Key 时使用本地抽取式回答）
python rag_platform.py ask "冰感内裤的面料是什么" --kb demo

# 3. 配置 LLM 生成（可选）
export DEEPSEEK_API_KEY=sk-xxx
python rag_platform.py ask "售后政策是什么" --kb demo

# 4. 评测回归
python eval.py --kb demo
```

## 目录结构

```
rag_platform.py   # 核心实现：解析/分块/向量化/检索/生成
eval.py           # 评测回归脚本
data/             # demo 语料
eval/evalset.jsonl# 评测集（50+ 条）
kb/               # 索引持久化（运行时生成）
```

## 技术栈

Python 3.10+（零依赖可运行核心链路）；可选：Chroma 向量库、DeepSeek API、Streamlit UI。

## 路线图

- [x] 文档解析与智能分块
- [x] 混合检索与引用溯源
- [x] 评测集与回归脚本
- [ ] OCR 图片文档解析
- [ ] Web UI（Streamlit）

## License

MIT
