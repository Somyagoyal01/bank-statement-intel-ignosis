"""End to end: synthetic statements -> categorise -> features -> flags -> results.

Usage: python -m bsi.run [--businesses 40]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from .categorize import Categorizer, score
from .features import business_features, flags
from .synth import HELDOUT, generate_portfolio


def main(businesses: int = 40, out: str = "results") -> dict:
    train = generate_portfolio(businesses, seed=1)
    test = generate_portfolio(businesses, seed=2)  # different businesses, names, references

    variants = {
        "model_only": Categorizer(use_rules=False, min_conf=0.0),
        "rules_plus_model": Categorizer(use_rules=True, min_conf=0.0),
        "rules_model_review(conf<0.7)": Categorizer(use_rules=True, min_conf=0.7),
    }
    cat_metrics = {}
    for name, c in variants.items():
        c.fit(train["narration"], train["direction"], train["category"])
        preds = c.predict(test["narration"], test["direction"], test["amount"])
        cat_metrics[name] = score(preds, test["category"])
    ho_t, ho_d, ho_y = map(list, zip(*HELDOUT))
    heldout = {name: score(c.predict(ho_t, ho_d, [0.0] * len(ho_t)), ho_y) for name, c in variants.items()}
    final = variants["rules_model_review(conf<0.7)"]
    preds = final.predict(test["narration"], test["direction"], test["amount"])
    test = test.assign(pred_category=[p for p, _, _ in preds])

    rows = []
    for bid, g in test.groupby("business_id"):
        f = business_features(g)
        rows.append({"business_id": bid, "true_stress": float(g["true_stress"].iloc[0]), **f, "flags": flags(f)})
    stressed = np.array([r["true_stress"] > 0 for r in rows])
    risk_score = np.array([len(r["flags"]) + r["bounces"] / 5 for r in rows])
    flagged = np.array([len(r["flags"]) > 0 for r in rows])
    results = {
        "test_transactions": len(test),
        "categorisation_synthetic_test": cat_metrics,
        "categorisation_heldout_formats": heldout,
        "underwriting_check": {
            "businesses": len(rows),
            "stressed_businesses": int(stressed.sum()),
            "flag_recall_on_stressed": round(float(flagged[stressed].mean()), 3) if stressed.any() else None,
            "flag_rate_on_healthy": round(float(flagged[~stressed].mean()), 3) if (~stressed).any() else None,
            "risk_score_auc": round(float(roc_auc_score(stressed, risk_score)), 3),
        },
        "example_business": rows[0],
    }
    Path(out).mkdir(exist_ok=True)
    Path(out, "results.json").write_text(json.dumps(results, indent=2, default=str))
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--businesses", type=int, default=40)
    print(json.dumps(main(ap.parse_args().businesses), indent=2, default=str))
