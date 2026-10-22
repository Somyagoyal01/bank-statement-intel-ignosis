"""Cash-flow features an MSME underwriter typically looks at, from categorised transactions.

These are generic, commonly published bank-statement-analysis checks (average
monthly business inflow, inflow volatility, existing EMI load, bounces, cash
share). Thresholds in `flags` are illustrative, not any lender's policy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def business_features(tx: pd.DataFrame, cat_col: str = "pred_category") -> dict:
    by_m = tx.groupby("month")
    inflow = by_m.apply(lambda g: g.loc[g[cat_col] == "customer_receipt", "amount"].sum(), include_groups=False)
    emi = by_m.apply(lambda g: g.loc[g[cat_col] == "loan_emi", "amount"].sum(), include_groups=False)
    months = max(tx["month"].nunique(), 1)
    credits = tx.loc[tx["direction"] == "CR", "amount"].sum()
    cash_dep = tx.loc[tx[cat_col] == "cash_deposit", "amount"].sum()
    trend = float(np.polyfit(inflow.index.to_numpy(), inflow.to_numpy(), 1)[0]) if len(inflow) > 1 else 0.0
    avg_in = float(inflow.mean())
    return {
        "avg_monthly_business_inflow": round(avg_in, 0),
        "inflow_cv": round(float(inflow.std(ddof=0) / avg_in), 3) if avg_in else None,
        "inflow_trend_per_month": round(trend, 0),
        "avg_monthly_emi": round(float(emi.mean()), 0),
        "emi_to_inflow": round(float(emi.mean() / avg_in), 3) if avg_in else None,
        "bounces": int((tx[cat_col] == "bounce_return").sum()),
        "bounces_per_month": round(float((tx[cat_col] == "bounce_return").sum() / months), 2),
        "cash_deposit_share_of_credits": round(float(cash_dep / credits), 3) if credits else None,
        "uncategorised_rows": int(tx[cat_col].isna().sum()),
    }


def flags(f: dict) -> list[str]:
    out = []
    if f["bounces_per_month"] >= 2.5:
        out.append(f"{f['bounces']} bounce/return entries ({f['bounces_per_month']}/month)")
    if f["emi_to_inflow"] is not None and f["emi_to_inflow"] > 0.5:
        out.append(f"EMI load {f['emi_to_inflow']:.0%} of business inflow")
    if f["inflow_cv"] is not None and f["inflow_cv"] > 0.6:
        out.append("volatile monthly inflow")
    if f["cash_deposit_share_of_credits"] and f["cash_deposit_share_of_credits"] > 0.4:
        out.append("high cash-deposit share")
    return out
