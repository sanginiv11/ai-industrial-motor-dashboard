"""ML ensemble integration slot (FUTURE-READY, currently DISABLED).

models/extra_trees.pkl and models/xgb.pkl expect 164 engineered features per
256-sample window (see models/RESULTS.txt). The code that produced those
features was not supplied, so the exact feature definitions/order are unknown.
Nothing here guesses them: predict() refuses to run until a real feature
extractor is provided.

To enable: implement extract_features(window_df) -> 1-D array of 164 values in
the training order, load both pickles, average predict_proba, and map argmax to
the class order in models/README.json. Then set ENABLED = True.
"""
import json, os
BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
ENABLED = False

def _read(name):
    p = os.path.join(BASE, name)
    return open(p, encoding="utf-8").read() if os.path.exists(p) else None

def status():
    readme = _read("README.json")
    return {
        "enabled": ENABLED,
        "reason": "Feature-extraction code for the 164 input features was not provided; no predictions are produced.",
        "files_present": sorted(os.listdir(BASE)) if os.path.isdir(BASE) else [],
        "readme": json.loads(readme) if readme else None,
        "results_text": _read("RESULTS.txt"),
    }

def extract_features(window_df):
    raise NotImplementedError("Feature extractor not supplied.")

def predict(window_df):
    if not ENABLED:
        raise RuntimeError("ML ensemble is not connected.")
    raise NotImplementedError
