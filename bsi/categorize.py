"""Transaction categorisation: rules -> ML model -> (optional) LLM, with confidence routing.

1. Rules catch unambiguous patterns (GST CPIN, NACH return, ATM withdrawal).
2. A char n-gram TF-IDF + logistic regression model handles the rest. Char n-grams
   cope with the abbreviations and random reference numbers in narrations.
3. Low-confidence rows go to an LLM (pluggable) or, by default, to a human-review queue.
   The offline default `FakeLLM` returns None (abstain), so nothing is guessed silently.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .synth import CATEGORIES

# Rules use general Indian banking vocabulary, written from public format knowledge.
RULES = [
    (re.compile(r"\b(RET|RTN|RETURN|BOUNCE|FAIL)\b.*|INSUFF", re.I), "bounce_return"),
    (re.compile(r"\bGST\b|CPIN|\bCIN\b|TDS|CHALLAN|ITNS|TAX PAYMENT", re.I), "gst_tax"),
    (re.compile(r"^(ATW|NWD)-|CASH WDL|CASH WITHDRAWAL|ATM CASH|ATM WDL", re.I), "cash_withdrawal"),
    (re.compile(r"CASH DEP|CSH DEP|BY CASH|CDM", re.I), "cash_deposit"),
    (re.compile(r"\bSELF\b|OWN A/?C|OWN ACCOUNT|SWEEP", re.I), "self_transfer"),
    (re.compile(r"\bSALARY\b|\bWAGES\b|SAL PAYOUT|PAYROLL", re.I), "salary"),
    (re.compile(r"\bNACH\b|\bACH D|\bECS\b|\bEMI\b|\bLOAN\b", re.I), "loan_emi"),
    (re.compile(r"ELECTRICITY|BROADBAND|\bRENT\b|BILLPAY|BILL PAID", re.I), "utility_rent"),
]


def _clean(text: str) -> str:
    # numbers carry no category signal; collapse them so the model can't memorise reference IDs
    return re.sub(r"\d+", "#", text.upper())


def apply_rules(text: str) -> Optional[str]:
    for pat, cat in RULES:
        if pat.search(text):
            return cat
    return None


class FakeLLM:
    """Offline stand-in. A real implementation would prompt an LLM with the narration,
    direction, amount and the allowed category list, and parse a JSON answer."""

    def __call__(self, narration: str, direction: str, amount: float) -> Optional[str]:
        return None  # abstain -> human review


@dataclass
class Categorizer:
    min_conf: float = 0.7
    use_rules: bool = True
    llm: Callable[[str, str, float], Optional[str]] = field(default_factory=FakeLLM)
    pipe: Optional[Pipeline] = None

    def fit(self, narrations, directions, labels) -> "Categorizer":
        X = [f"{d} {_clean(t)}" for t, d in zip(narrations, directions)]
        self.pipe = Pipeline([
            ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)),
            ("clf", LogisticRegression(max_iter=3000, C=3.0)),
        ]).fit(X, labels)
        return self

    def predict(self, narrations, directions, amounts):
        """Return list of (category or None, source, confidence). None = needs human review."""
        X = [f"{d} {_clean(t)}" for t, d in zip(narrations, directions)]
        proba = self.pipe.predict_proba(X)
        classes = self.pipe.classes_
        out = []
        for t, d, a, p in zip(narrations, directions, amounts, proba):
            if self.use_rules and (r := apply_rules(t)):
                out.append((r, "rule", 1.0))
                continue
            conf, cat = float(p.max()), classes[int(p.argmax())]
            if conf >= self.min_conf:
                out.append((cat, "model", conf))
                continue
            llm_cat = self.llm(t, d, a)
            if llm_cat in CATEGORIES:
                out.append((llm_cat, "llm", conf))
            else:
                out.append((None, "review", conf))
        return out


def score(preds, labels) -> dict:
    labels = list(labels)
    auto = [(p, l) for (p, _, _), l in zip(preds, labels) if p is not None]
    acc = float(np.mean([p == l for p, l in auto])) if auto else 0.0
    return {
        "auto_categorised_share": round(len(auto) / max(len(labels), 1), 3),
        "auto_accuracy": round(acc, 4),
        "review_queue_share": round(1 - len(auto) / max(len(labels), 1), 3),
        "errors_auto": int(sum(p != l for p, l in auto)),
    }
