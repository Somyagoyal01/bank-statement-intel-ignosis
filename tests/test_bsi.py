import pandas as pd

from bsi.categorize import Categorizer, apply_rules, score
from bsi.features import business_features, flags
from bsi.run import main
from bsi.synth import CATEGORIES, generate_portfolio, generate_statement


def test_synth_labels_and_directions():
    df = generate_statement("B1", months=3, seed=3)
    assert set(df["category"]) <= set(CATEGORIES)
    assert set(df["direction"]) <= {"CR", "DR"}
    assert (df.loc[df["category"] == "customer_receipt", "direction"] == "CR").all()


def test_rules():
    assert apply_rules("GST PAYMENT CPIN 12345678901234") == "gst_tax"
    assert apply_rules("NACH RTN BAJAJ FIN FUNDS INSUFFICIENT") == "bounce_return"
    assert apply_rules("ATW-123456-HDFC ATM-PUNE") == "cash_withdrawal"
    assert apply_rules("UPI/CR/1234/SAI PACKAGING/HDFC/x@okaxis/PAYMENT") is None


def test_categoriser_generalises_to_new_businesses():
    train, test = generate_portfolio(15, seed=1), generate_portfolio(10, seed=2)
    c = Categorizer(min_conf=0.0).fit(train["narration"], train["direction"], train["category"])
    s = score(c.predict(test["narration"], test["direction"], test["amount"]), test["category"])
    assert s["auto_accuracy"] > 0.95


def test_low_confidence_goes_to_review_not_guess():
    c = Categorizer(min_conf=0.99, use_rules=False)
    train = generate_portfolio(5, seed=4)
    c.fit(train["narration"], train["direction"], train["category"])
    preds = c.predict(["XYZ UNKNOWN 123"], ["DR"], [100.0])
    assert preds[0][0] is None and preds[0][1] == "review"


def test_llm_hook_used_when_unsure():
    c = Categorizer(min_conf=0.99, use_rules=False, llm=lambda t, d, a: "utility_rent")
    train = generate_portfolio(5, seed=4)
    c.fit(train["narration"], train["direction"], train["category"])
    assert c.predict(["XYZ UNKNOWN"], ["DR"], [1.0])[0][:2] == ("utility_rent", "llm")


def test_features_and_flags():
    tx = pd.DataFrame({
        "month": [1, 1, 2, 2, 2], "direction": ["CR", "DR", "CR", "DR", "DR"],
        "amount": [100000, 60000, 50000, 60000, 590],
        "pred_category": ["customer_receipt", "loan_emi", "customer_receipt", "loan_emi", "bounce_return"],
    })
    f = business_features(tx)
    assert f["avg_monthly_business_inflow"] == 75000 and f["bounces"] == 1
    assert any("EMI load" in x for x in flags(f))
    assert not any("bounce" in x for x in flags(f))  # 0.5 bounces/month is below the threshold


def test_end_to_end(tmp_path):
    res = main(businesses=12, out=str(tmp_path))
    assert (tmp_path / "results.json").exists()
    assert res["categorisation_synthetic_test"]["rules_plus_model"]["auto_accuracy"] > 0.95
