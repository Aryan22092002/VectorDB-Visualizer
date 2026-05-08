"""
VectorDB Engine — Python port of main.cpp
Requires: pip install flask requests numpy
Optional: Ollama running locally (https://ollama.com)
Run:      python main.py
"""

from __future__ import annotations

import json
import math
import heapq
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional
import requests
from flask import Flask, request, jsonify, Response
import numpy as np

# =====================================================================
#  CONSTANTS
# =====================================================================

DIMS = 16  # demo vectors (fixed dimension)

# =====================================================================
#  DATA TYPES
# =====================================================================

@dataclass
class VectorItem:
    id: int
    metadata: str
    category: str
    emb: list[float]


DistFn = Callable[[list[float], list[float]], float]

# =====================================================================
#  DISTANCE METRICS
# =====================================================================

def euclidean(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na  = sum(x * x for x in a)
    nb  = sum(y * y for y in b)
    if na < 1e-9 or nb < 1e-9:
        return 1.0
    return 1.0 - dot / (math.sqrt(na) * math.sqrt(nb))


def manhattan(a: list[float], b: list[float]) -> float:
    return sum(abs(x - y) for x, y in zip(a, b))


def get_dist_fn(metric: str) -> DistFn:
    if metric == "cosine":
        return cosine
    if metric == "manhattan":
        return manhattan
    return euclidean

# =====================================================================
#  BRUTE FORCE
# =====================================================================

class BruteForce:
    def __init__(self):
        self.items: list[VectorItem] = []

    def insert(self, v: VectorItem):
        self.items.append(v)

    def knn(self, q: list[float], k: int, dist: DistFn) -> list[tuple[float, int]]:
        scored = [(dist(q, v.emb), v.id) for v in self.items]
        scored.sort()
        return scored[:k]

    def remove(self, id_: int):
        self.items = [v for v in self.items if v.id != id_]

# =====================================================================
#  KD-TREE
# =====================================================================

class KDNode:
    def __init__(self, item: VectorItem):
        self.item  = item
        self.left:  Optional[KDNode] = None
        self.right: Optional[KDNode] = None


class KDTree:
    def __init__(self, dims: int):
        self.dims = dims
        self.root: Optional[KDNode] = None

    def _insert(self, node: Optional[KDNode], v: VectorItem, depth: int) -> KDNode:
        if node is None:
            return KDNode(v)
        ax = depth % self.dims
        if v.emb[ax] < node.item.emb[ax]:
            node.left  = self._insert(node.left,  v, depth + 1)
        else:
            node.right = self._insert(node.right, v, depth + 1)
        return node

    def insert(self, v: VectorItem):
        self.root = self._insert(self.root, v, 0)

    def _knn(self, node: Optional[KDNode], q: list[float], k: int,
             depth: int, dist: DistFn, heap: list):
        if node is None:
            return
        dn = dist(q, node.item.emb)
        # Max-heap stored as negated values
        if len(heap) < k or dn < -heap[0][0]:
            heapq.heappush(heap, (-dn, node.item.id))
            if len(heap) > k:
                heapq.heappop(heap)
        ax   = depth % self.dims
        diff = q[ax] - node.item.emb[ax]
        closer  = node.left  if diff < 0 else node.right
        farther = node.right if diff < 0 else node.left
        self._knn(closer,  q, k, depth + 1, dist, heap)
        if len(heap) < k or abs(diff) < -heap[0][0]:
            self._knn(farther, q, k, depth + 1, dist, heap)

    def knn(self, q: list[float], k: int, dist: DistFn) -> list[tuple[float, int]]:
        heap: list = []
        self._knn(self.root, q, k, 0, dist, heap)
        return sorted(((-neg_d, id_) for neg_d, id_ in heap))

    def rebuild(self, items: list[VectorItem]):
        self.root = None
        for v in items:
            self.insert(v)

# =====================================================================
#  HNSW — Hierarchical Navigable Small World
# =====================================================================

class HNSW:
    @dataclass
    class Node:
        item:    VectorItem
        max_lyr: int
        nbrs:    list[list[int]] = field(default_factory=list)

    def __init__(self, M: int = 16, ef_build: int = 200):
        self.M        = M
        self.M0       = 2 * M
        self.ef_build = ef_build
        self.mL       = 1.0 / math.log(float(M))
        self.G:       dict[int, HNSW.Node] = {}
        self.top_layer = -1
        self.entry_pt  = -1
        random.seed(42)

    def _rand_level(self) -> int:
        return int(math.floor(-math.log(random.random()) * self.mL))

    def _search_layer(self, q: list[float], ep: int, ef: int,
                      lyr: int, dist: DistFn) -> list[tuple[float, int]]:
        visited: set[int] = set()
        d0 = dist(q, self.G[ep].item.emb)
        visited.add(ep)

        # Min-heap for candidates
        cands = [(d0, ep)]
        # Max-heap for found (negate)
        found = [(-d0, ep)]

        while cands:
            cd, cid = heapq.heappop(cands)
            if len(found) >= ef and cd > -found[0][0]:
                break
            node = self.G.get(cid)
            if node is None or lyr >= len(node.nbrs):
                continue
            for nid in node.nbrs[lyr]:
                if nid in visited or nid not in self.G:
                    continue
                visited.add(nid)
                nd = dist(q, self.G[nid].item.emb)
                if len(found) < ef or nd < -found[0][0]:
                    heapq.heappush(cands, (nd, nid))
                    heapq.heappush(found, (-nd, nid))
                    if len(found) > ef:
                        heapq.heappop(found)

        result = sorted((-neg_d, id_) for neg_d, id_ in found)
        return result

    def _select_nbrs(self, cands: list[tuple[float, int]], max_m: int) -> list[int]:
        return [id_ for _, id_ in cands[:max_m]]

    def insert(self, item: VectorItem, dist: DistFn):
        id_  = item.id
        lvl  = self._rand_level()
        node = HNSW.Node(item=item, max_lyr=lvl, nbrs=[[] for _ in range(lvl + 1)])
        self.G[id_] = node

        if self.entry_pt == -1:
            self.entry_pt  = id_
            self.top_layer = lvl
            return

        ep = self.entry_pt
        for lc in range(self.top_layer, lvl, -1):
            if lc < len(self.G[ep].nbrs):
                W = self._search_layer(item.emb, ep, 1, lc, dist)
                if W:
                    ep = W[0][1]

        for lc in range(min(self.top_layer, lvl), -1, -1):
            W    = self._search_layer(item.emb, ep, self.ef_build, lc, dist)
            maxM = self.M0 if lc == 0 else self.M
            sel  = self._select_nbrs(W, maxM)
            node.nbrs[lc] = sel

            for nid in sel:
                if nid not in self.G:
                    continue
                nbr_node = self.G[nid]
                if len(nbr_node.nbrs) <= lc:
                    nbr_node.nbrs.extend([] for _ in range(lc + 1 - len(nbr_node.nbrs)))
                nbr_node.nbrs[lc].append(id_)
                if len(nbr_node.nbrs[lc]) > maxM:
                    ds = [(dist(nbr_node.item.emb, self.G[c].item.emb), c)
                          for c in nbr_node.nbrs[lc] if c in self.G]
                    ds.sort()
                    nbr_node.nbrs[lc] = [c for _, c in ds[:maxM]]

            if W:
                ep = W[0][1]

        if lvl > self.top_layer:
            self.top_layer = lvl
            self.entry_pt  = id_

    def knn(self, q: list[float], k: int, ef: int,
            dist: DistFn) -> list[tuple[float, int]]:
        if self.entry_pt == -1:
            return []
        ep = self.entry_pt
        for lc in range(self.top_layer, 0, -1):
            if lc < len(self.G[ep].nbrs):
                W = self._search_layer(q, ep, 1, lc, dist)
                if W:
                    ep = W[0][1]
        W = self._search_layer(q, ep, max(ef, k), 0, dist)
        return W[:k]

    def remove(self, id_: int):
        if id_ not in self.G:
            return
        for nid, nd in self.G.items():
            for layer in nd.nbrs:
                if id_ in layer:
                    layer.remove(id_)
        if self.entry_pt == id_:
            self.entry_pt = next((nid for nid in self.G if nid != id_), -1)
        del self.G[id_]

    def get_info(self) -> dict:
        top_layer  = self.top_layer
        node_count = len(self.G)
        max_l      = max(top_layer + 1, 1)
        nodes_per_layer = [0] * max_l
        edges_per_layer = [0] * max_l
        nodes_out: list[dict] = []
        edges_out: list[dict] = []

        for id_, nd in self.G.items():
            nodes_out.append({"id": id_, "metadata": nd.item.metadata,
                               "category": nd.item.category, "maxLyr": nd.max_lyr})
            for lc in range(min(nd.max_lyr + 1, max_l)):
                nodes_per_layer[lc] += 1
                if lc < len(nd.nbrs):
                    for nid in nd.nbrs[lc]:
                        if id_ < nid:
                            edges_per_layer[lc] += 1
                            edges_out.append({"src": id_, "dst": nid, "lyr": lc})

        return {"topLayer": top_layer, "nodeCount": node_count,
                "nodesPerLayer": nodes_per_layer, "edgesPerLayer": edges_per_layer,
                "nodes": nodes_out, "edges": edges_out}

    def __len__(self):
        return len(self.G)

# =====================================================================
#  VECTOR DATABASE  (demo 16D index)
# =====================================================================

class VectorDB:
    def __init__(self, dims: int):
        self.dims   = dims
        self.store:  dict[int, VectorItem] = {}
        self.bf      = BruteForce()
        self.kdt     = KDTree(dims)
        self.hnsw    = HNSW(16, 200)
        self.lock    = threading.Lock()
        self._next_id = 1

    def insert(self, meta: str, cat: str, emb: list[float], dist: DistFn) -> int:
        with self.lock:
            v = VectorItem(id=self._next_id, metadata=meta, category=cat, emb=emb)
            self._next_id += 1
            self.store[v.id] = v
            self.bf.insert(v)
            self.kdt.insert(v)
            self.hnsw.insert(v, dist)
            return v.id

    def remove(self, id_: int) -> bool:
        with self.lock:
            if id_ not in self.store:
                return False
            del self.store[id_]
            self.bf.remove(id_)
            self.hnsw.remove(id_)
            self.kdt.rebuild(list(self.store.values()))
            return True

    def search(self, q: list[float], k: int, metric: str, algo: str) -> dict:
        with self.lock:
            dfn = get_dist_fn(metric)
            t0  = time.perf_counter()
            if algo == "bruteforce":
                raw = self.bf.knn(q, k, dfn)
            elif algo == "kdtree":
                raw = self.kdt.knn(q, k, dfn)
            else:
                raw = self.hnsw.knn(q, k, 50, dfn)
            us = int((time.perf_counter() - t0) * 1_000_000)

            hits = []
            for d, id_ in raw:
                if id_ in self.store:
                    v = self.store[id_]
                    hits.append({"id": v.id, "meta": v.metadata,
                                 "cat": v.category, "emb": v.emb, "dist": d})
            return {"hits": hits, "us": us, "algo": algo, "metric": metric}

    def benchmark(self, q: list[float], k: int, metric: str) -> dict:
        with self.lock:
            dfn = get_dist_fn(metric)
            def time_fn(fn):
                t = time.perf_counter()
                fn()
                return int((time.perf_counter() - t) * 1_000_000)
            return {
                "bfUs":   time_fn(lambda: self.bf.knn(q, k, dfn)),
                "kdUs":   time_fn(lambda: self.kdt.knn(q, k, dfn)),
                "hnswUs": time_fn(lambda: self.hnsw.knn(q, k, 50, dfn)),
                "n":      len(self.store),
            }

    def all(self) -> list[VectorItem]:
        with self.lock:
            return list(self.store.values())

    def hnsw_info(self) -> dict:
        with self.lock:
            return self.hnsw.get_info()

    def size(self) -> int:
        with self.lock:
            return len(self.store)

# =====================================================================
#  TEXT CHUNKER
# =====================================================================

def chunk_text(text: str, chunk_words: int = 250, overlap_words: int = 30) -> list[str]:
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_words:
        return [text]
    chunks = []
    step   = chunk_words - overlap_words
    i = 0
    while i < len(words):
        end   = min(i + chunk_words, len(words))
        chunk = " ".join(words[i:end])
        chunks.append(chunk)
        if end == len(words):
            break
        i += step
    return chunks

# =====================================================================
#  OLLAMA CLIENT
# =====================================================================

class OllamaClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 11434):
        self.base    = f"http://{host}:{port}"
        self.embed_model = "nomic-embed-text"
        self.gen_model   = "llama3.2"
        self._resolved_gen_model: Optional[str] = None

    def is_available(self) -> bool:
        try:
            r = requests.get(f"{self.base}/api/tags", timeout=2)
            return r.status_code == 200
        except Exception:
            return False

    def _parse_embedding_payload(self, data: object) -> list[float]:
        if not isinstance(data, dict):
            return []
        if isinstance(data.get("error"), str) and data["error"]:
            return []
        # OpenAI-compatible: POST /v1/embeddings
        blk = data.get("data")
        if isinstance(blk, list) and blk:
            row = blk[0]
            vec = row.get("embedding") if isinstance(row, dict) else None
            if isinstance(vec, list) and vec and isinstance(vec[0], (int, float)):
                return vec
        embs = data.get("embeddings")
        if isinstance(embs, list) and embs:
            head = embs[0]
            if isinstance(head, list) and head and isinstance(head[0], (int, float)):
                return head
            if isinstance(head, (int, float)):
                return embs
        one = data.get("embedding")
        if isinstance(one, list) and one:
            if isinstance(one[0], list):
                inn = one[0]
                if inn and isinstance(inn[0], (int, float)):
                    return inn
            elif isinstance(one[0], (int, float)):
                return one
        return []

    def embed(self, text: str) -> list[float]:
        t = (text or "").strip()
        if not t:
            return []
        attempts = [
            (f"{self.base}/api/embed", {"model": self.embed_model, "input": t}),
            (f"{self.base}/api/embed", {"model": self.embed_model, "input": [t]}),
            (f"{self.base}/api/embeddings", {"model": self.embed_model, "prompt": t}),
            (f"{self.base}/v1/embeddings", {"model": self.embed_model, "input": t}),
        ]
        for url, payload in attempts:
            try:
                r = requests.post(url, json=payload, timeout=60)
                if r.status_code != 200:
                    continue
                vec = self._parse_embedding_payload(r.json())
                if vec:
                    return vec
            except Exception:
                continue
        return []

    def list_generation_models(self) -> list[str]:
        try:
            r = requests.get(f"{self.base}/api/tags", timeout=6)
            if r.status_code != 200:
                return []
            out: list[str] = []
            for m in r.json().get("models") or []:
                name = m.get("name")
                if not name:
                    continue
                details = m.get("details") or {}
                fams = details.get("families")
                if isinstance(fams, list) and fams == ["nomic-bert"]:
                    continue
                out.append(name)
            out.sort(key=lambda n: (0 if "llama" in n.lower() else 1, n.lower()))
            return out
        except Exception:
            return []

    def generate(self, prompt: str) -> str:
        order: list[str] = []
        if self._resolved_gen_model:
            order.append(self._resolved_gen_model)
        order.append(self.gen_model)
        for cand in self.list_generation_models():
            if cand not in order:
                order.append(cand)

        last_err = ""
        for model in order:
            try:
                r = requests.post(
                    f"{self.base}/api/generate",
                    json={"model": model, "prompt": prompt, "stream": False},
                    timeout=183,
                )
                try:
                    data = r.json()
                except Exception:
                    data = {}
                msg = ""
                if isinstance(data, dict):
                    msg = (data.get("error") or data.get("message") or "").strip()
                txt = ""
                if isinstance(data, dict):
                    txt = (data.get("response") or "").strip()
                if r.status_code == 200 and txt:
                    self._resolved_gen_model = model
                    return txt
                if msg:
                    last_err = msg
                elif not r.ok:
                    last_err = str(r.status_code)
            except Exception as e:
                last_err = str(e)
                continue

        if "not found" in last_err.lower():
            last_err += " Pull a chat model with: ollama pull llama3.2:1b (or llama3.2 / llama3)"
        elif "does not support generate" in last_err.lower():
            last_err += " Pick a Llama/Gemma/etc. chat model, not an embeddings-only model."

        hint = ""
        avail = ", ".join(self.list_generation_models()[:12])
        if avail:
            hint = f" Installed chat-capable models: {avail}"
        return f"ERROR: No working generation model ({last_err or 'could not reach Ollama'}).{hint}"

    def cached_generation_model(self) -> Optional[str]:
        return self._resolved_gen_model

# =====================================================================
#  DOCUMENT DATABASE
# =====================================================================

@dataclass
class DocItem:
    id:    int
    title: str
    text:  str
    emb:   list[float]


class DocumentDB:
    def __init__(self):
        self.store:  dict[int, DocItem] = {}
        self.hnsw    = HNSW(16, 200)
        self.bf      = BruteForce()
        self.lock    = threading.Lock()
        self._next_id = 1
        self._dims   = 0

    def insert(self, title: str, text: str, emb: list[float]) -> int:
        with self.lock:
            if self._dims == 0:
                self._dims = len(emb)
            item = DocItem(id=self._next_id, title=title, text=text, emb=emb)
            self._next_id += 1
            self.store[item.id] = item
            vi = VectorItem(id=item.id, metadata=title, category="doc", emb=emb)
            self.hnsw.insert(vi, cosine)
            self.bf.insert(vi)
            return item.id

    def search(self, q: list[float], k: int) -> list[tuple[float, DocItem]]:
        """Top-k chunks by cosine distance (lower = more similar)."""
        with self.lock:
            if not self.store:
                return []
            raw = (self.bf.knn(q, k, cosine)
                   if len(self.store) < 10
                   else self.hnsw.knn(q, k, 50, cosine))
            return [(d, self.store[id_])
                    for d, id_ in raw
                    if id_ in self.store]

    def remove(self, id_: int) -> bool:
        with self.lock:
            if id_ not in self.store:
                return False
            del self.store[id_]
            self.hnsw.remove(id_)
            self.bf.remove(id_)
            return True

    def all(self) -> list[DocItem]:
        with self.lock:
            return list(self.store.values())

    def size(self) -> int:
        with self.lock:
            return len(self.store)

    def get_dims(self) -> int:
        return self._dims

# =====================================================================
#  DEMO DATA  (16D categorical vectors)
# =====================================================================

def load_demo(db: VectorDB):
    dist = get_dist_fn("cosine")
    rows = [
        ("Linked List: nodes connected by pointers", "cs",
         [0.90,0.85,0.72,0.68,0.12,0.08,0.15,0.10,0.05,0.08,0.06,0.09,0.07,0.11,0.08,0.06]),
        ("Binary Search Tree: O(log n) search and insert", "cs",
         [0.88,0.82,0.78,0.74,0.15,0.10,0.08,0.12,0.06,0.07,0.08,0.05,0.09,0.06,0.07,0.10]),
        ("Dynamic Programming: memoization overlapping subproblems", "cs",
         [0.82,0.76,0.88,0.80,0.20,0.18,0.12,0.09,0.07,0.06,0.08,0.07,0.08,0.09,0.06,0.07]),
        ("Graph BFS and DFS: breadth and depth first traversal", "cs",
         [0.85,0.80,0.75,0.82,0.18,0.14,0.10,0.08,0.06,0.09,0.07,0.06,0.10,0.08,0.09,0.07]),
        ("Hash Table: O(1) lookup with collision chaining", "cs",
         [0.87,0.78,0.70,0.76,0.13,0.11,0.09,0.14,0.08,0.07,0.06,0.08,0.07,0.10,0.08,0.09]),
        ("Calculus: derivatives integrals and limits", "math",
         [0.12,0.15,0.18,0.10,0.91,0.86,0.78,0.72,0.08,0.06,0.07,0.09,0.07,0.08,0.06,0.10]),
        ("Linear Algebra: matrices eigenvalues eigenvectors", "math",
         [0.20,0.18,0.15,0.12,0.88,0.90,0.82,0.76,0.09,0.07,0.08,0.06,0.10,0.07,0.08,0.09]),
        ("Probability: distributions random variables Bayes theorem", "math",
         [0.15,0.12,0.20,0.18,0.84,0.80,0.88,0.82,0.07,0.08,0.06,0.10,0.09,0.06,0.09,0.08]),
        ("Number Theory: primes modular arithmetic RSA cryptography", "math",
         [0.22,0.16,0.14,0.20,0.80,0.85,0.76,0.90,0.08,0.09,0.07,0.06,0.08,0.10,0.07,0.06]),
        ("Combinatorics: permutations combinations generating functions", "math",
         [0.18,0.20,0.16,0.14,0.86,0.78,0.84,0.80,0.06,0.07,0.09,0.08,0.06,0.09,0.10,0.07]),
        ("Neapolitan Pizza: wood-fired dough San Marzano tomatoes", "food",
         [0.08,0.06,0.09,0.07,0.07,0.08,0.06,0.09,0.90,0.86,0.78,0.72,0.08,0.06,0.09,0.07]),
        ("Sushi: vinegared rice raw fish and nori rolls", "food",
         [0.06,0.08,0.07,0.09,0.09,0.06,0.08,0.07,0.86,0.90,0.82,0.76,0.07,0.09,0.06,0.08]),
        ("Ramen: noodle soup with chashu pork and soft-boiled eggs", "food",
         [0.09,0.07,0.06,0.08,0.08,0.09,0.07,0.06,0.82,0.78,0.90,0.84,0.09,0.07,0.08,0.06]),
        ("Tacos: corn tortillas with carnitas salsa and cilantro", "food",
         [0.07,0.09,0.08,0.06,0.06,0.07,0.09,0.08,0.78,0.82,0.86,0.90,0.06,0.08,0.07,0.09]),
        ("Croissant: laminated pastry with buttery flaky layers", "food",
         [0.06,0.07,0.10,0.09,0.10,0.06,0.07,0.10,0.85,0.80,0.76,0.82,0.09,0.07,0.10,0.06]),
        ("Basketball: fast-paced shooting dribbling slam dunks", "sports",
         [0.09,0.07,0.08,0.10,0.08,0.09,0.07,0.06,0.08,0.07,0.09,0.06,0.91,0.85,0.78,0.72]),
        ("Football: tackles touchdowns field goals and strategy", "sports",
         [0.07,0.09,0.06,0.08,0.09,0.07,0.10,0.08,0.07,0.09,0.08,0.07,0.87,0.89,0.82,0.76]),
        ("Tennis: racket volleys groundstrokes and Wimbledon serves", "sports",
         [0.08,0.06,0.09,0.07,0.07,0.08,0.06,0.09,0.09,0.06,0.07,0.08,0.83,0.80,0.88,0.82]),
        ("Chess: openings endgames tactics strategic board game", "sports",
         [0.25,0.20,0.22,0.18,0.22,0.18,0.20,0.15,0.06,0.08,0.07,0.09,0.80,0.84,0.78,0.90]),
        ("Swimming: butterfly freestyle backstroke Olympic competition", "sports",
         [0.06,0.08,0.07,0.09,0.08,0.06,0.09,0.07,0.10,0.08,0.06,0.07,0.85,0.82,0.86,0.80]),
    ]
    for meta, cat, emb in rows:
        db.insert(meta, cat, emb, dist)

# =====================================================================
#  FLASK HTTP SERVER
# =====================================================================

app    = Flask(__name__)
db     = VectorDB(DIMS)
doc_db = DocumentDB()
ollama = OllamaClient()


def cors_headers(response: Response) -> Response:
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


@app.after_request
def add_cors(response: Response) -> Response:
    return cors_headers(response)


@app.route("/", methods=["OPTIONS"])
@app.route("/<path:path>", methods=["OPTIONS"])
def preflight(path=""):
    return Response(status=204)


# ── DEMO VECTOR ENDPOINTS ─────────────────────────────────────────────

@app.route("/search")
def search():
    v_str  = request.args.get("v", "")
    q      = [float(x) for x in v_str.split(",") if x]
    if len(q) != DIMS:
        return jsonify({"error": f"need {DIMS}D vector"}), 400

    k      = int(request.args.get("k",      5))
    metric = request.args.get("metric", "cosine")
    algo   = request.args.get("algo",   "hnsw")

    out = db.search(q, k, metric, algo)
    results = [
        {"id": h["id"], "metadata": h["meta"], "category": h["cat"],
         "distance": round(h["dist"], 6), "embedding": h["emb"]}
        for h in out["hits"]
    ]
    return jsonify({"results": results, "latencyUs": out["us"],
                    "algo": out["algo"], "metric": out["metric"]})


@app.route("/insert", methods=["POST"])
def insert():
    body = request.get_json(force=True) or {}
    meta = body.get("metadata", "")
    cat  = body.get("category", "")
    emb  = body.get("embedding", [])
    if not meta or len(emb) != DIMS:
        return jsonify({"error": "invalid body"}), 400
    id_ = db.insert(meta, cat, emb, get_dist_fn("cosine"))
    return jsonify({"id": id_})


@app.route("/delete/<int:id_>", methods=["DELETE"])
def delete_item(id_: int):
    ok = db.remove(id_)
    return jsonify({"ok": ok})


@app.route("/items")
def items():
    return jsonify([
        {"id": v.id, "metadata": v.metadata,
         "category": v.category, "embedding": v.emb}
        for v in db.all()
    ])


@app.route("/benchmark")
def benchmark():
    v_str  = request.args.get("v", "")
    q      = [float(x) for x in v_str.split(",") if x]
    if len(q) != DIMS:
        return jsonify({"error": f"need {DIMS}D vector"}), 400
    k      = int(request.args.get("k", 5))
    metric = request.args.get("metric", "cosine")
    b      = db.benchmark(q, k, metric)
    return jsonify({"bruteforceUs": b["bfUs"], "kdtreeUs": b["kdUs"],
                    "hnswUs": b["hnswUs"], "itemCount": b["n"]})


@app.route("/hnsw-info")
def hnsw_info():
    return jsonify(db.hnsw_info())


# ── DOCUMENT + RAG ENDPOINTS ──────────────────────────────────────────

@app.route("/doc/insert", methods=["POST"])
def doc_insert():
    body  = request.get_json(force=True) or {}
    title = body.get("title", "")
    text  = body.get("text",  "")
    if not title or not text:
        return jsonify({"error": "need title and text"}), 400

    chunks = chunk_text(text, 250, 30)
    ids    = []
    for i, chunk in enumerate(chunks):
        emb = ollama.embed(chunk)
        if not emb:
            return jsonify({
                "error": "Ollama unavailable. Install from https://ollama.com then run: "
                         "ollama pull nomic-embed-text && ollama pull llama3.2"
            }), 503
        chunk_title = (f"{title} [{i+1}/{len(chunks)}]"
                       if len(chunks) > 1 else title)
        ids.append(doc_db.insert(chunk_title, chunk, emb))

    return jsonify({"ids": ids, "chunks": len(chunks), "dims": doc_db.get_dims()})


@app.route("/doc/delete/<int:id_>", methods=["DELETE"])
def doc_delete(id_: int):
    ok = doc_db.remove(id_)
    return jsonify({"ok": ok})


@app.route("/doc/list")
def doc_list():
    docs = doc_db.all()
    result = []
    for d in docs:
        preview = d.text[:120] + ("…" if len(d.text) > 120 else "")
        result.append({"id": d.id, "title": d.title,
                        "preview": preview, "words": len(d.text.split())})
    return jsonify(result)


@app.route("/doc/search", methods=["POST"])
def doc_search():
    body     = request.get_json(force=True) or {}
    question = body.get("question", "")
    k        = int(body.get("k", 3))
    if not question:
        return jsonify({"error": "need question"}), 400

    q_emb = ollama.embed(question)
    if not q_emb:
        return jsonify({"error": "Ollama unavailable"}), 503

    hits = doc_db.search(q_emb, k)
    contexts = [{"id": item.id, "title": item.title,
                 "distance": round(d, 4)} for d, item in hits]
    return jsonify({"contexts": contexts})


@app.route("/doc/ask", methods=["POST"])
def doc_ask():
    body     = request.get_json(force=True) or {}
    question = body.get("question", "")
    k        = int(body.get("k", 3))
    if not question:
        return jsonify({"error": "need question"}), 400

    # Step 1: embed question
    q_emb = ollama.embed(question)
    if not q_emb:
        return jsonify({"error": "Ollama unavailable"}), 503

    # Step 2: retrieve top-k chunks
    hits = doc_db.search(q_emb, k)

    # Step 3: build prompt
    ctx_parts = [f"[{i+1}] {item.title}:\n{item.text}\n"
                 for i, (_, item) in enumerate(hits)]
    ctx_str   = "\n".join(ctx_parts)
    prompt    = (
        "You are a helpful assistant. Answer the user's question directly. "
        "Use the provided context if it contains relevant information. "
        "If it doesn't, just use your own general knowledge. "
        "IMPORTANT: Do NOT mention the 'context', 'provided text', or say things like "
        "'the context doesn't mention'. Just answer the question naturally.\n\n"
        f"Context:\n{ctx_str}\nQuestion: {question}\n\nAnswer:"
    )

    # Step 4: generate answer
    answer = ollama.generate(prompt)

    # Step 5: return everything
    contexts = [{"id": item.id, "title": item.title,
                 "text": item.text, "distance": round(d, 4)}
                for d, item in hits]
    return jsonify({"answer": answer, "model": ollama.gen_model,
                    "contexts": contexts, "docCount": doc_db.size()})


@app.route("/status")
def status():
    up = ollama.is_available()
    gens = ollama.list_generation_models() if up else []
    return jsonify({
        "ollamaAvailable": up,
        "embedModel":      ollama.embed_model,
        "genModel":        ollama.gen_model,
        "generationModels": gens,
        "activeGenModel":  ollama.cached_generation_model(),
        "docCount":        doc_db.size(),
        "docDims":         doc_db.get_dims(),
        "demoDims":        DIMS,
        "demoCount":       db.size(),
    })


@app.route("/stats")
def stats():
    return jsonify({
        "count":      db.size(),
        "dims":       DIMS,
        "algorithms": ["bruteforce", "kdtree", "hnsw"],
        "metrics":    ["euclidean", "cosine", "manhattan"],
    })


@app.route("/")
def index():
    try:
        with open("index.html") as f:
            return Response(f.read(), mimetype="text/html")
    except FileNotFoundError:
        return Response("index.html not found", status=404)


# =====================================================================
#  MAIN
# =====================================================================

if __name__ == "__main__":
    load_demo(db)

    ollama_up = ollama.is_available()
    print("=== VectorDB Engine ===")
    print("http://localhost:8080")
    print(f"{db.size()} demo vectors | {DIMS} dims | HNSW+KD-Tree+BruteForce")
    print(f"Ollama: {'ONLINE' if ollama_up else 'OFFLINE (install from ollama.com)'}")
    if ollama_up:
        print(f"  embed model: {ollama.embed_model}  gen model: {ollama.gen_model}")

    app.run(host="0.0.0.0", port=8080, threaded=True)