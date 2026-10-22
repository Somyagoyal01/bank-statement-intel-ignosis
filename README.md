# Bank-statement intelligence for MSME underwriting: categorise, route doubts, flag risk

> Personal project by Somya Goyal, built while applying to Ignosis. Not affiliated with or endorsed by Ignosis; uses public/synthetic data.

## Why this matters for Ignosis

- **Problem:** Indian bank-statement narrations are messy: UPI/NEFT/NACH strings, a different format per bank, random reference numbers. A categoriser that is 95% right still puts wrong numbers into an underwriting decision unless it knows *when it is unsure*.
- **Who feels it:** credit and underwriting teams at lenders using account-aggregator data for MSME loans. Ignosis lists bank-statement analytics and MSME underwriting among its products ([ignosis.ai](https://ignosis.ai/careers/ml-engineer); [funding announcement](https://ignosis.ai/blog/ignosis-raises-4m-led-by-peak-xvs-surge-to-democratise-financial-data-inclusion-in-india-with-dpi-rails-and-ai-driven-analytics)).
- **What this shows:** a rules → ML → (LLM or human) pipeline with confidence routing, tested on narration formats it never saw in training. It turns categorised transactions into the cash-flow features an underwriter reads: inflow, volatility, EMI load, bounces and cash share.

## Results (from `python -m bsi.run`; train and test use different synthetic businesses)

**Categorisation on 30 hand-written narrations in unseen bank formats** (SBI "BY/TO TRANSFER", "MB:", POS, MMT/IMPS, payroll and more):

| Pipeline | Auto-categorised | Accuracy on auto rows | Sent to review |
|---|---|---|---|
| Char n-gram TF-IDF + logistic regression only | 100% | 90.0% (3 errors) | 0% |
| Rules + model | 100% | 96.7% (1 error) | 0% |
| Rules + model + review when confidence < 0.7 | 90% | **100%** (0 errors) | 10% |

On the synthetic test set (same formats as training, 12,231 transactions), every variant scores 100%. That's why I don't trust in-distribution numbers here: the unseen-format test is the one that matters. I wrote the rules and the held-out narrations myself, so the 96.7% and 100% figures are optimistic. The 90% model-only figure is the cleanest one.

**Underwriting check (40 synthetic businesses, 17 with injected stress: fewer receipts, more bounces):** flags caught 12 of 17 stressed businesses (70.6% recall) with 0 false flags on healthy ones. Stress is injected by a known rule in the generator, so this is a sanity check of the plumbing, not a credit-model claim.

## How it works

- `bsi/synth.py`: a labelled synthetic generator for 10 categories (customer receipt, vendor payment, salary, GST/tax, loan EMI, cash in/out, bounce/return, utility/rent, self-transfer). Names, VPAs and references are random, and there is a `stress` knob per business. It also holds `HELDOUT`, the unseen-format test set.
- `bsi/categorize.py`:
  1. Regex rules for unambiguous patterns (CPIN/challan, NACH return, ATM, payroll).
  2. A char n-gram model on direction + narration with digits masked, so it can't memorise reference numbers.
  3. Low-confidence rows go to `llm(narration, direction, amount)`. The default `FakeLLM` abstains, which sends the row to human review, so nothing is guessed silently.
- `bsi/features.py`: per-business features (average monthly business inflow, inflow CV, trend, EMI-to-inflow, bounces per month, cash-deposit share, uncategorised rows) and illustrative flags. Thresholds are examples, not any lender's policy.
- `bsi/run.py`: runs end to end and writes `results/results.json`.

## Run it

```bash
pip install -r requirements.txt
python -m bsi.run --businesses 40
pytest -q            # 7 tests
```

**Real LLM path (optional):** pass `llm=my_fn` to `Categorizer`. `my_fn` should prompt a model with the narration, direction, amount and the 10 allowed labels, ask for JSON `{"category": ...}`, and return `None` when unsure. Keep PII out of prompts, or use a self-hosted model.

## Limitations / next steps

- Synthetic narrations are cleaner than real ones. The next step is a public or consented sample, a larger held-out set per bank, and per-bank error analysis.
- Counterparty resolution (same customer under different names or VPAs) and related-party/circular-transaction detection would make the features much stronger.
- Calibrate the confidence threshold against review-team capacity: pick the threshold that keeps the review queue within N rows per day.
