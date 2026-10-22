"""Labelled SYNTHETIC Indian bank-statement transactions for small businesses (MSMEs).

Narrations imitate common public formats (UPI/NEFT/IMPS/NACH/ATM/cheque-return
strings). Every name, VPA and account fragment is randomly generated. No real
statements or customer data are used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FIRMS = ["SHREE GANESH TRADERS", "AARAV ENTERPRISES", "KRISHNA AGRO", "PATEL STEEL", "MAA DURGA TEXTILES",
         "SAI PACKAGING", "OM SALES CORP", "GLOBAL AUTO PARTS", "NEW INDIA HARDWARE", "RIDDHI PHARMA DIST"]
PEOPLE = ["RAMESH K", "SUNITA DEVI", "AMIT SHARMA", "PRIYA N", "VIKAS GUPTA", "ANJALI R", "MOHAN LAL", "FAROOQ A"]
BANKS = ["HDFC", "SBIN", "ICIC", "UTIB", "KKBK", "PUNB", "BARB"]
LENDERS = ["BAJAJ FIN", "TATA CAP", "LENDINGKART", "HDFC BANK LOAN", "IDFC FIRST"]

CATEGORIES = ["customer_receipt", "vendor_payment", "salary", "gst_tax", "loan_emi", "cash_withdrawal",
              "cash_deposit", "bounce_return", "utility_rent", "self_transfer"]


def _ref(rng, n=12):
    return "".join(rng.choice(list("0123456789"), n))


def narration(cat: str, rng) -> tuple[str, str]:
    """Return (narration, direction) for a category."""
    firm, person, bank = rng.choice(FIRMS), rng.choice(PEOPLE), rng.choice(BANKS)
    vpa = f"{person.split()[0].lower()}{rng.integers(10, 99)}@ok{rng.choice(['axis', 'hdfcbank', 'sbi', 'icici'])}"
    if cat == "customer_receipt":
        return rng.choice([f"UPI/CR/{_ref(rng)}/{firm}/{bank}/{vpa}/PAYMENT",
                           f"NEFT CR-{bank}0{_ref(rng, 6)}-{firm}-INV {rng.integers(100, 999)}",
                           f"IMPS/P2A/{_ref(rng)}/{firm}/SALE PROCEEDS"]), "CR"
    if cat == "vendor_payment":
        return rng.choice([f"UPI/DR/{_ref(rng)}/{firm}/{bank}/{vpa}/PURCHASE",
                           f"NEFT DR-{bank}0{_ref(rng, 6)}-{firm}-RAW MATERIAL",
                           f"RTGS DR-{bank}R{_ref(rng, 10)}-{firm}-SUPPLY BILL"]), "DR"
    if cat == "salary":
        return rng.choice([f"NEFT DR-{bank}0{_ref(rng, 6)}-{person}-SALARY {rng.choice(['JAN', 'FEB', 'MAR'])}",
                           f"IMPS/{_ref(rng)}/{person}/WAGES",
                           f"BULK SAL PAYOUT {_ref(rng, 6)} {rng.integers(5, 40)} EMP"]), "DR"
    if cat == "gst_tax":
        return rng.choice([f"GST PAYMENT CPIN {_ref(rng, 14)}", f"GSTN/{_ref(rng, 10)}/CGST SGST",
                           f"TDS CHALLAN 281 {_ref(rng, 8)}", f"ADV TAX ITNS280 {_ref(rng, 8)}"]), "DR"
    if cat == "loan_emi":
        lender = rng.choice(LENDERS)
        return rng.choice([f"NACH DR {lender} {_ref(rng, 8)} EMI", f"ACH D- {lender}-{_ref(rng, 10)}",
                           f"ECS/{lender}/LOAN A/C {_ref(rng, 6)}"]), "DR"
    if cat == "cash_withdrawal":
        return rng.choice([f"ATW-{_ref(rng, 6)}-{bank} ATM-{rng.choice(['JAIPUR', 'SURAT', 'PUNE'])}",
                           f"CASH WDL SELF CHQ {_ref(rng, 6)}", f"NWD-{_ref(rng, 8)}-ATM CASH"]), "DR"
    if cat == "cash_deposit":
        return rng.choice([f"CASH DEP {rng.choice(['BRANCH', 'CDM'])} {_ref(rng, 6)}",
                           f"BY CASH-{rng.choice(['JAIPUR', 'SURAT', 'PUNE'])} BR", "CDM DEPOSIT SELF"]), "CR"
    if cat == "bounce_return":
        return rng.choice([f"CHQ RET {_ref(rng, 6)} INSUFF FUNDS", f"NACH RTN {rng.choice(LENDERS)} FUNDS INSUFFICIENT",
                           f"I/W CHQ RETURN CHGS {_ref(rng, 6)}", f"ECS RETURN CHARGES {_ref(rng, 6)}"]), "DR"
    if cat == "utility_rent":
        return rng.choice([f"BILLPAY/{rng.choice(['MSEDCL', 'JVVNL', 'BESCOM', 'TORRENT PWR'])}/{_ref(rng, 8)}",
                           f"UPI/DR/{_ref(rng)}/JIO FIBER/BROADBAND", f"NEFT DR-{bank}0{_ref(rng, 6)}-{person}-SHOP RENT"]), "DR"
    if cat == "self_transfer":
        return rng.choice([f"IMPS/{_ref(rng)}/SELF/OWN A/C TRF", f"FT-SELF-{bank} {_ref(rng, 6)} SWEEP",
                           f"UPI/DR/{_ref(rng)}/OWN ACCOUNT/{bank}"]), rng.choice(["DR", "CR"])
    raise ValueError(cat)


AMOUNT = {  # (median, spread) in INR
    "customer_receipt": (45000, 0.9), "vendor_payment": (38000, 0.9), "salary": (18000, 0.5),
    "gst_tax": (12000, 0.8), "loan_emi": (21000, 0.3), "cash_withdrawal": (8000, 0.7),
    "cash_deposit": (15000, 0.8), "bounce_return": (590, 0.3), "utility_rent": (6500, 0.6),
    "self_transfer": (25000, 0.9),
}
MIX = [0.31, 0.22, 0.10, 0.05, 0.05, 0.08, 0.06, 0.01, 0.07, 0.05]


def generate_statement(business_id: str, months: int = 6, seed: int = 0, stress: float = 0.0) -> pd.DataFrame:
    """One business's statement. `stress` (0..1) lowers receipts and adds bounces."""
    rng = np.random.default_rng(seed)
    mix = np.array(MIX)
    mix[0] *= 1 - 0.4 * stress
    mix[7] *= 1 + 6 * stress
    mix = mix / mix.sum()
    rows = []
    for m in range(months):
        for _ in range(rng.integers(35, 70)):
            cat = rng.choice(CATEGORIES, p=mix)
            text, direction = narration(cat, rng)
            med, sd = AMOUNT[cat]
            amt = float(np.round(med * rng.lognormal(0, sd) * (1 - 0.3 * stress * (cat == "customer_receipt")), 2))
            rows.append({"business_id": business_id, "month": m + 1, "day": int(rng.integers(1, 29)),
                         "narration": text, "direction": direction, "amount": amt, "category": cat})
    return pd.DataFrame(rows).sort_values(["month", "day"]).reset_index(drop=True)


def generate_portfolio(n_businesses: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(n_businesses):
        stress = float(rng.choice([0.0, 0.0, 0.0, 0.5, 1.0]))
        df = generate_statement(f"B{i:03d}", seed=seed * 1000 + i, stress=stress)
        frames.append(df.assign(true_stress=stress))
    return pd.concat(frames, ignore_index=True)


# Hand-written narrations in OTHER banks' formats (SBI "BY/TO TRANSFER", Kotak "MB:",
# POS, MMT/IMPS etc.). Never used for training; they test generalisation to unseen formats.
HELDOUT = [
    ("BY TRANSFER-UPI/CR/604512345678/MEHTA TRADING CO/SBIN/mehta@oksbi/UPI--", "CR", "customer_receipt"),
    ("BY TRANSFER-NEFT*HDFC0001234*N123456*KAVYA EXPORTS--", "CR", "customer_receipt"),
    ("MB:RECEIVED FROM GUPTA FABRICS INV 3321", "CR", "customer_receipt"),
    ("MMT/IMPS/604512340001/PAYMENT FOR ORDER/SONI ELECTRICALS/KKBK", "CR", "customer_receipt"),
    ("TO TRANSFER-UPI/DR/604598765432/JAI AMBE SUPPLIERS/BARB/jai@ybl/PAY--", "DR", "vendor_payment"),
    ("MB:SENT TO BHARAT CHEMICALS PO 778", "DR", "vendor_payment"),
    ("TO TRANSFER-RTGS UTR NO: SBINR52026102200 VINAYAK METALS", "DR", "vendor_payment"),
    ("POS 4591XXXXXXXX1234 RELIANCE DIGITAL", "DR", "vendor_payment"),
    ("TO TRANSFER-NEFT SALARY OCT RAHUL VERMA", "DR", "salary"),
    ("STAFF PAYROLL OCT 2026 18 EMPLOYEES", "DR", "salary"),
    ("IMPS-P2A-604500000-SURESH-WAGES OCT", "DR", "salary"),
    ("GST PMT-CIN 26102200123456", "DR", "gst_tax"),
    ("TAX PAYMENT-TDS-CHALLAN NO 00123", "DR", "gst_tax"),
    ("ICCL GST CHALLAN 2610220012", "DR", "gst_tax"),
    ("ACH DEBIT-HDFC BANK LTD-LOAN 84512", "DR", "loan_emi"),
    ("NACH-DR-FULLERTON-EMI-OCT", "DR", "loan_emi"),
    ("EMI 12 OF 36 PERSONAL LOAN 5521", "DR", "loan_emi"),
    ("ATM WDL-ATM CASH 1234 MG ROAD", "DR", "cash_withdrawal"),
    ("CASH WITHDRAWAL BY CHEQUE 000123", "DR", "cash_withdrawal"),
    ("CSH DEP (CDM) 0012 MAIN BRANCH", "CR", "cash_deposit"),
    ("CASH DEPOSIT BY SELF", "CR", "cash_deposit"),
    ("RETURN-ACH DEBIT-INSUFFICIENT BALANCE", "DR", "bounce_return"),
    ("CHQ BOUNCE CHARGES 000456 + GST", "DR", "bounce_return"),
    ("NACH FAIL CHARGES OCT", "DR", "bounce_return"),
    ("BILL PAID-BESCOM ELECTRICITY 445566", "DR", "utility_rent"),
    ("TO TRANSFER-UPI/DR/6045/AIRTEL BROADBAND/AIRP", "DR", "utility_rent"),
    ("RENT OCT GODOWN - LANDLORD SHARMA", "DR", "utility_rent"),
    ("TRANSFER TO OWN ACCOUNT 5521XXXX", "DR", "self_transfer"),
    ("FUND TRF FROM CURRENT A/C SELF", "CR", "self_transfer"),
    ("SWEEP IN FD 0034", "CR", "self_transfer"),
]
