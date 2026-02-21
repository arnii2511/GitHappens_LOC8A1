from importlib import import_module

__all__ = ["HybridRanker"]


def __getattr__(name: str):
    if name == "HybridRanker":
        return import_module(".hybrid_ranker", __name__).HybridRanker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
