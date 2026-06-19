#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RAG Knowledge Platform - 轻量级 RAG 问答平台（对标 RAGFlow/FastGPT 核心链路）。

用法:
    python rag_platform.py ingest <目录或文件> --kb demo
    python rag_platform.py ask "<问题>" --kb demo
"""
import argparse, csv, json, math, os, re, sys, zlib
from collections import Counter
from pathlib import Path

STOP = set("的了和与及或在是有对于为这那你我它上下中".split())

# ---------------- 1. 文档解析 ----------------
class DocumentParser:
    SUPPORTED = {".txt", ".md", ".csv"}

    def parse(self, path: Path):
        if path.suffix.lower() not in self.SUPPORTED:
            return None
        text = path.read_text(encoding="utf-8", errors="ignore")
        if path.suffix.lower() == ".csv":
            rows = ["/".join(r) for r in csv.reader(text.splitlines()) if r]
            return {"doc": path.name, "text": "\n".join(rows), "title": path.stem}
        title = path.stem
        # markdown 保留标题层级
        text = re.sub(r"^(#{1,6})\s*(.+)$", r"【H\1】\2", text, flags=re.M)
        return {"doc": path.name, "text": text, "title": title}

# ---------------- 2. 智能分块 ----------------
class Chunker:
    def __init__(self, size=200, overlap=50):
        self.size, self.overlap = size, overlap

    def split(self, doc):
        text, chunks, start, idx = doc["text"], [], 0, 0
        while start < len(text):
            piece = text[start:start + self.size].strip()
            if piece:
                heading = self._last_heading(doc["text"][:start])
                chunks.append({"doc": doc["doc"], "cid": idx, "text": piece, "section": heading})
                idx += 1
            start += self.size - self.overlap
        return chunks

    @staticmethod
    def _last_heading(prefix):
        heads = re.findall(r"【H#+】[^\n]+", prefix)
        return heads[-1][:30] if heads else ""

# ---------------- 3. 向量化（本地 TF-IDF，可切 API）----------------
class Embedder:
    def __init__(self, dim=512):
        self.dim, self.idf = dim, {}

    @staticmethod
    def tokens(text):
        # 中文按二元词组(bigram)切分，英文/数字按词切分
        words = []
        for tok in re.findall(r"[a-zA-Z0-9]+|[\u4e00-\u9fa5]+", text.lower()):
            if re.fullmatch(r"[\u4e00-\u9fa5]+", tok):
                if len(tok) == 1:
                    if tok not in STOP:
                        words.append(tok)
                else:
                    words.extend(tok[i:i+2] for i in range(len(tok) - 1))
            elif tok not in STOP:
                words.append(tok)
        return words

    def fit(self, chunks):
        df = Counter()
        for c in chunks:
            df.update(set(self.tokens(c["text"])))
        n = max(len(chunks), 1)
        self.idf = {w: math.log((n + 1) / (c + 1)) + 1 for w, c in df.items()}

    def embed(self, text):
        tf = Counter(self.tokens(text))
        v = [0.0] * self.dim
        for w, c in tf.items():
            if w in self.idf:
                h = zlib.crc32(w.encode("utf-8")) & 0xFFFFFFFF
                v[h % self.dim] += self.idf[w] * (1 + math.log(c))
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

# ---------------- 4. 索引（多知识库隔离）----------------
class VectorStore:
    def __init__(self, root="kb"):
        self.root = Path(root)

    def path(self, kb):
        return self.root / kb / "index.json"

    def save(self, kb, chunks, vecs):
        p = self.path(kb)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"chunks": chunks, "vecs": vecs}, ensure_ascii=False), encoding="utf-8")

    def load(self, kb):
        p = self.path(kb)
        if not p.exists():
            sys.exit(f"知识库 [{kb}] 不存在，先执行 ingest")
        d = json.loads(p.read_text(encoding="utf-8"))
        return d["chunks"], d["vecs"]

# ---------------- 5. 混合检索 ----------------
class HybridRetriever:
    def __init__(self, chunks, vecs, embedder, topk=4, kw_weight=0.3):
        self.chunks, self.vecs, self.emb = chunks, vecs, embedder
        self.topk, self.kw_weight = topk, kw_weight

    @staticmethod
    def cosine(a, b):
        return sum(x * y for x, y in zip(a, b))

    def search(self, query):
        qv = self.emb.embed(query)
        qtokens = set(Embedder.tokens(query))
        scored = []
        for c, v in zip(self.chunks, self.vecs):
            vs = self.cosine(qv, v)
            ctokens = set(Embedder.tokens(c["text"]))
            ks = len(qtokens & ctokens) / (len(qtokens) or 1)
            scored.append((0.7 * vs + self.kw_weight * ks, c))
        scored.sort(key=lambda x: -x[0])
        return scored[: self.topk]

# ---------------- 6. 生成（带引用，可选 LLM）----------------
class Generator:
    def __init__(self, api_key=None, base_url="https://api.deepseek.com/v1"):
        self.key, self.url = api_key, base_url

    def answer(self, query, hits):
        context = "\n\n".join(f"[{i+1}] ({h['doc']}#{h['cid']} {h['section']}) {h['text']}" for i, (_, h) in enumerate(hits))
        if not self.key:
            best = max(hits, key=lambda x: len(x[1]["text"]))[1]
            return f"（抽取式回答）{best['text'][:120]}\n\n来源: {best['doc']}#{best['cid']}"
        try:
            import requests
            r = requests.post(f"{self.url}/chat/completions", timeout=30, headers={
                "Authorization": f"Bearer {self.key}", "Content-Type": "application/json"},
                json={"model": "deepseek-chat", "messages": [
                    {"role": "system", "content": "仅依据以下资料回答，用中文，末尾列出引用编号。"},
                    {"role": "user", "content": f"资料:\n{context}\n\n问题: {query}"}]})
            return r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"[LLM 调用失败，降级为抽取式] {e}"

# ---------------- CLI ----------------
def cmd_ingest(args):
    src = Path(args.source)
    files = [src] if src.is_file() else [f for f in src.rglob("*") if f.suffix.lower() in DocumentParser.SUPPORTED]
    parser, chunker, emb = DocumentParser(), Chunker(), Embedder()
    chunks = []
    for f in files:
        doc = parser.parse(f)
        if doc:
            chunks.extend(chunker.split(doc))
    emb.fit(chunks)
    vecs = [emb.embed(c["text"]) for c in chunks]
    VectorStore().save(args.kb, chunks, vecs)
    print(f"ingest 完成: {len(files)} 个文件 -> {len(chunks)} 个分块, 知识库[{args.kb}]")

def cmd_ask(args):
    chunks, vecs = VectorStore().load(args.kb)
    emb = Embedder(); emb.fit(chunks)
    hits = HybridRetriever(chunks, vecs, emb, topk=args.topk).search(args.query)
    for score, h in hits:
        print(f"  [hit {score:.3f}] {h['doc']}#{h['cid']} {h['section']}")
    print("\n" + Generator(api_key=os.environ.get("DEEPSEEK_API_KEY")).answer(args.query, hits))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="RAG Knowledge Platform")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("ingest"); p1.add_argument("source"); p1.add_argument("--kb", default="default"); p1.set_defaults(func=cmd_ingest)
    p2 = sub.add_parser("ask"); p2.add_argument("query"); p2.add_argument("--kb", default="default"); p2.add_argument("--topk", type=int, default=4); p2.set_defaults(func=cmd_ask)
    a = ap.parse_args(); a.func(a)
