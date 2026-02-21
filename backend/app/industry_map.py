from __future__ import annotations


CANONICAL = {
    "medical devices": "healthcare",
    "pharmaceuticals": "healthcare",
    "pharma": "healthcare",
    "healthcare": "healthcare",
    "chemicals": "chemicals",
    "machinery": "engineering",
    "engineering": "engineering",
    "auto parts": "automotive",
    "automotive": "automotive",
    "electronics": "electronics",
    "electricals": "electronics",
    "it software": "it",
    "software": "it",
    "it": "it",
    "solar": "energy",
    "energy": "energy",
    "textiles": "textiles",
}


RELATED = {
    "healthcare": {"healthcare": 1.0, "chemicals": 0.4},
    "engineering": {"engineering": 1.0, "automotive": 0.7, "electronics": 0.5},
    "automotive": {"automotive": 1.0, "engineering": 0.7},
    "electronics": {"electronics": 1.0, "engineering": 0.5, "it": 0.4, "energy": 0.4},
    "it": {"it": 1.0, "electronics": 0.4},
    "energy": {"energy": 1.0, "electronics": 0.4},
    "textiles": {"textiles": 1.0},
    "chemicals": {"chemicals": 1.0, "healthcare": 0.4},
}


def canonicalize(industry: str) -> str:
    if not isinstance(industry, str):
        return "unknown"
    key = industry.strip().lower()
    if key == "":
        return "unknown"
    return CANONICAL.get(key, key)


def industry_similarity(buyer_industry: str, exporter_industry: str) -> float:
    b = canonicalize(buyer_industry)
    e = canonicalize(exporter_industry)
    if b == "unknown" or e == "unknown":
        return 0.0
    if b == e:
        return 1.0
    return float(RELATED.get(b, {}).get(e, 0.0))
