from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..industry_map import RELATED, canonicalize


@dataclass
class IndustryRule:
    buyer_cluster: str
    exporter_cluster: str
    support: float
    confidence: float
    lift: float
    score: float


class IndustryAssociationMiner:
    def __init__(
        self,
        min_support: float = 0.0005,
        min_confidence: float = 0.60,
        min_lift: float = 1.20,
        max_links_per_industry: int = 3,
        semantic_backfill_top_k: int = 2,
        semantic_backfill_min_sim: float = 0.6,
    ):
        self.min_support = float(max(0.0, min_support))
        self.min_confidence = float(max(0.0, min_confidence))
        self.min_lift = float(max(0.0, min_lift))
        self.max_links_per_industry = int(max(1, max_links_per_industry))
        self.semantic_backfill_top_k = int(max(0, semantic_backfill_top_k))
        self.semantic_backfill_min_sim = float(np.clip(semantic_backfill_min_sim, 0.0, 1.0))
        self.rules: dict[str, list[IndustryRule]] = {}
        self.ready = False

    def fit(self, pairs: list[tuple[str, str]]):
        self.rules = {}
        self.ready = False
        if not pairs:
            return

        buyer_counts: dict[str, int] = {}
        exporter_counts: dict[str, int] = {}
        pair_counts: dict[tuple[str, str], int] = {}
        total = 0

        for b_raw, e_raw in pairs:
            b = canonicalize(b_raw)
            e = canonicalize(e_raw)
            if b == "unknown" or e == "unknown":
                continue
            total += 1
            buyer_counts[b] = int(buyer_counts.get(b, 0) + 1)
            exporter_counts[e] = int(exporter_counts.get(e, 0) + 1)
            key = (b, e)
            pair_counts[key] = int(pair_counts.get(key, 0) + 1)

        if total <= 0:
            return

        grouped: dict[str, list[IndustryRule]] = {}
        for (b, e), cnt in pair_counts.items():
            if b == e:
                continue
            support = float(cnt / total)
            conf = float(cnt / max(1, buyer_counts.get(b, 1)))
            p_e = float(exporter_counts.get(e, 0) / total)
            lift = float(conf / max(1e-9, p_e))
            if support < self.min_support or conf < self.min_confidence or lift < self.min_lift:
                continue
            score = float(conf * np.log1p(lift))
            grouped.setdefault(b, []).append(
                IndustryRule(
                    buyer_cluster=b,
                    exporter_cluster=e,
                    support=support,
                    confidence=conf,
                    lift=lift,
                    score=score,
                )
            )

        for b, rules in grouped.items():
            rules_sorted = sorted(rules, key=lambda r: (r.score, r.confidence, r.lift), reverse=True)
            self.rules[b] = rules_sorted[: self.max_links_per_industry]

        self.ready = bool(self.rules)

    def associated_exporter_clusters(self, buyer_industry: str) -> dict[str, float]:
        b = canonicalize(buyer_industry)
        if b == "unknown":
            return {}
        rules = self.rules.get(b, [])
        if rules:
            top = rules[: self.max_links_per_industry]
            return {r.exporter_cluster: float(r.score) for r in top}

        # Controlled fallback for sparse history: only top few high-sim related industries.
        if self.semantic_backfill_top_k <= 0:
            return {}
        rel = []
        for e, sim in RELATED.get(b, {}).items():
            if e == b:
                continue
            s = float(np.clip(sim, 0.0, 1.0))
            if s < self.semantic_backfill_min_sim:
                continue
            rel.append((e, s))
        if not rel:
            return {}
        rel.sort(key=lambda x: x[1], reverse=True)
        rel = rel[: self.semantic_backfill_top_k]
        return {e: s for e, s in rel}

    def associated_exporter_list(self, buyer_industry: str) -> list[str]:
        return list(self.associated_exporter_clusters(buyer_industry).keys())
