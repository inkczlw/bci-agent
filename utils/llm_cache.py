"""LLM 调用缓存。

应用层缓存策略：对相同或相似的 query 复用已有的 LLM 响应，
减少重复 API 调用和 token 消耗。

注意：这不是 Anthropic/OpenAI 的原生 prompt cache（那是在 API 层面
缓存 prefix），而是应用层的语义缓存 + 精确匹配缓存。


- 两层缓存设计：L1 精确匹配（hash）+ L2 语义匹配（embedding similarity）
- 为什么不只用精确匹配："Neuralink 的技术" 和 "告诉我 Neuralink 的核心技术"
  语义相同但字面不同，精确匹配命中不了
- 为什么不只用语义匹配：embedding 计算本身有成本，高频重复 query 直接 hash 查更快
- TTL 机制：避免缓存过期数据（新闻类查询 TTL 短，公司基本信息 TTL 长）
"""

import hashlib
import json
import time
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("bci_agent.cache")


class LLMCache:
    """两层 LLM 响应缓存：精确匹配 + 语义匹配。"""

    def __init__(
        self,
        cache_dir: str = "cache",
        default_ttl: int = 3600,      # 默认 1 小时
        similarity_threshold: float = 0.85,
        embedding_fn=None,
    ):
        """
        Args:
            cache_dir: 缓存文件目录
            default_ttl: 默认过期时间（秒）
            similarity_threshold: 语义匹配的相似度阈值
            embedding_fn: 用于语义缓存的 embedding 函数。
                         None 则只启用 L1（精确匹配）。
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.default_ttl = default_ttl
        self.similarity_threshold = similarity_threshold
        self.embedding_fn = embedding_fn

        # L1: 内存中的精确匹配缓存 {hash: {"response": ..., "expires_at": ...}}
        self._exact_cache: dict[str, dict] = {}

        # L2: 语义缓存的 embedding 索引 [(embedding, hash, expires_at), ...]
        self._semantic_index: list[tuple[list[float], str, float]] = []

        # 统计
        self.stats = {"l1_hits": 0, "l2_hits": 0, "misses": 0, "evictions": 0}

        # 加载持久化缓存
        self._load_from_disk()

    def get(self, query: str, tool_context: str = "") -> dict | None:
        """查找缓存。

        Args:
            query: 用户 query
            tool_context: tool 调用上下文（用于区分同 query 不同 tool 路径）

        Returns:
            {"response": ..., "cache_level": "L1"|"L2"} or None
        """
        cache_key = self._make_key(query, tool_context)

        # L1: 精确匹配
        if cache_key in self._exact_cache:
            entry = self._exact_cache[cache_key]
            if time.time() < entry["expires_at"]:
                self.stats["l1_hits"] += 1
                logger.debug(f"Cache L1 hit: {query[:50]}...")
                return {"response": entry["response"], "cache_level": "L1"}
            else:
                # 过期了，清理
                del self._exact_cache[cache_key]
                self.stats["evictions"] += 1

        # L2: 语义匹配（如果有 embedding 函数）
        if self.embedding_fn is not None:
            result = self._semantic_lookup(query, cache_key)
            if result is not None:
                self.stats["l2_hits"] += 1
                logger.debug(f"Cache L2 hit: {query[:50]}...")
                return result

        self.stats["misses"] += 1
        return None

    def put(self, query: str, response: Any, tool_context: str = "", ttl: int | None = None):
        """写入缓存。"""
        cache_key = self._make_key(query, tool_context)
        expires_at = time.time() + (ttl or self.default_ttl)

        # L1: 精确匹配
        self._exact_cache[cache_key] = {
            "response": response,
            "expires_at": expires_at,
            "query": query,
            "created_at": time.time(),
        }

        # L2: 语义索引
        if self.embedding_fn is not None:
            try:
                embedding = self.embedding_fn(query)
                self._semantic_index.append((embedding, cache_key, expires_at))
            except Exception as e:
                logger.warning(f"Failed to compute embedding for cache: {e}")

        # 持久化
        self._save_to_disk(cache_key)

    def invalidate(self, query: str = None, tool_context: str = ""):
        """手动失效缓存条目。"""
        if query:
            cache_key = self._make_key(query, tool_context)
            self._exact_cache.pop(cache_key, None)
            self._semantic_index = [
                (emb, k, exp) for emb, k, exp in self._semantic_index if k != cache_key
            ]
        else:
            # 全部清空
            self._exact_cache.clear()
            self._semantic_index.clear()
            logger.info("Cache fully invalidated")

    def get_stats(self) -> dict:
        """返回缓存统计。"""
        total = self.stats["l1_hits"] + self.stats["l2_hits"] + self.stats["misses"]
        hit_rate = (
            round((self.stats["l1_hits"] + self.stats["l2_hits"]) / total * 100, 2)
            if total > 0 else 0.0
        )
        return {
            **self.stats,
            "total_queries": total,
            "hit_rate_pct": hit_rate,
            "cache_size": len(self._exact_cache),
            "semantic_index_size": len(self._semantic_index),
        }

    def cleanup_expired(self):
        """清理过期条目。"""
        now = time.time()
        expired_keys = [k for k, v in self._exact_cache.items() if now >= v["expires_at"]]
        for k in expired_keys:
            del self._exact_cache[k]
            self.stats["evictions"] += 1

        self._semantic_index = [
            (emb, k, exp) for emb, k, exp in self._semantic_index if now < exp
        ]

        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")

    # ── 内部方法 ──────────────────────────────────────────────

    def _make_key(self, query: str, tool_context: str = "") -> str:
        """生成缓存 key：query + tool_context 的 hash。"""
        raw = f"{query.strip().lower()}|{tool_context}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _semantic_lookup(self, query: str, exclude_key: str = "") -> dict | None:
        """语义相似度查找。"""
        try:
            query_emb = self.embedding_fn(query)
        except Exception:
            return None

        now = time.time()
        best_score = 0.0
        best_key = None

        for emb, cache_key, expires_at in self._semantic_index:
            if expires_at <= now or cache_key == exclude_key:
                continue
            score = self._cosine_similarity(query_emb, emb)
            if score > best_score:
                best_score = score
                best_key = cache_key

        if best_score >= self.similarity_threshold and best_key in self._exact_cache:
            return {
                "response": self._exact_cache[best_key]["response"],
                "cache_level": "L2",
                "similarity": round(best_score, 4),
            }
        return None

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """计算两个向量的余弦相似度。"""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _save_to_disk(self, cache_key: str):
        """持久化单个缓存条目。"""
        if cache_key not in self._exact_cache:
            return
        entry = self._exact_cache[cache_key]
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump({
                    "query": entry.get("query", ""),
                    "response": str(entry["response"]),  # 转 string 保证可序列化
                    "expires_at": entry["expires_at"],
                    "created_at": entry.get("created_at", time.time()),
                }, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Failed to persist cache entry: {e}")

    def _load_from_disk(self):
        """从磁盘加载缓存。"""
        now = time.time()
        loaded = 0
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    entry = json.load(f)
                if entry.get("expires_at", 0) > now:
                    key = cache_file.stem
                    self._exact_cache[key] = entry
                    loaded += 1
                else:
                    cache_file.unlink()  # 删除过期文件
            except Exception:
                continue
        if loaded:
            logger.info(f"Loaded {loaded} cache entries from disk")