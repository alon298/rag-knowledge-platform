#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""评测回归：对 eval/evalset.jsonl 跑召回命中率与回答包含率。"""
import argparse, json
from pathlib import Path
from rag_platform import DocumentParser, Chunker, Embedder, VectorStore, HybridRetriever

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", default="demo")
    ap.add_argument("--topk", type=int, default=4)
    args = ap.parse_args()
    chunks, vecs = VectorStore().load(args.kb)
    emb = Embedder(); emb.fit(chunks)
    ret = HybridRetriever(chunks, vecs, emb, topk=args.topk)
    cases = [json.loads(l) for l in Path("eval/evalset.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    hit = contain = 0
    for c in cases:
        hits = ret.search(c["q"])
        found = any(c["gold_doc"] == h["doc"] and c.get("gold_cid") in (None, h["cid"]) for _, h in hits)
        hit += found
        contain += any(c["answer_key"] in h["text"] for _, h in hits)
    n = len(cases)
    print(f"用例数: {n}\nchunk 召回命中率: {hit/n:.1%}\n关键词包含率: {contain/n:.1%}")

if __name__ == "__main__":
    main()
