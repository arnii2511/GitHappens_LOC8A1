import json
import importlib.util


def main():
    report = {
        "xgboost_available": False,
        "xgboost_cuda_ready": False,
        "cupy_available": False,
        "torch_available": False,
        "torch_cuda_available": False,
        "sentence_transformers_available": False,
    }

    try:
        import xgboost as xgb

        report["xgboost_available"] = True
        try:
            _ = xgb.XGBClassifier(tree_method="hist", device="cuda")
            report["xgboost_cuda_ready"] = True
        except Exception:
            report["xgboost_cuda_ready"] = False
    except Exception:
        pass

    try:
        import cupy as cp

        _ = cp.cuda.runtime.getDeviceCount()
        report["cupy_available"] = True
    except Exception:
        report["cupy_available"] = False

    try:
        import torch

        report["torch_available"] = True
        report["torch_cuda_available"] = bool(torch.cuda.is_available())
    except Exception:
        report["torch_available"] = False
        report["torch_cuda_available"] = False

    report["sentence_transformers_available"] = bool(importlib.util.find_spec("sentence_transformers"))

    report["full_gpu_ready"] = bool(
        report["xgboost_cuda_ready"] and report["cupy_available"] and report["torch_cuda_available"]
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
