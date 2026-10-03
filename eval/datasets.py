"""Synthetic eval datasets, each with one planted answer (fixed seeds -> same CSVs every time).

Run:  python -m eval.datasets   (run_eval.py also creates any missing files)

| File            | Planted ground truth                                                   |
|-----------------|------------------------------------------------------------------------|
| churn.csv       | Basic plan churns ~28%; Standard and Premium ~11%                       |
| orders.csv      | Electronics has the highest average order value; Home has most orders   |
| deliveries.csv  | October delivery times rise only for carrier FastShip (+3 days)         |
| discounts.csv   | Units sold rise with discount_pct (positive association)                |
| conversions.csv | Redesign in week 7 does NOT change the conversion rate (null case)      |
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"


def make_churn(rng) -> pd.DataFrame:
    plans = {"Basic": (9.99, 0.30), "Standard": (19.99, 0.10), "Premium": (29.99, 0.11)}
    rows = []
    for i in range(1200):
        plan = rng.choice(list(plans), p=[0.4, 0.35, 0.25])
        fee, churn_prob = plans[plan]
        rows.append({"customer_id": f"C{i:04d}", "plan": plan, "monthly_fee": fee,
                     "tenure_months": int(rng.integers(1, 37)),
                     "churned": int(rng.random() < churn_prob)})
    return pd.DataFrame(rows)


def make_orders(rng) -> pd.DataFrame:
    # (share of orders, mean order value): Home is the most common, Electronics the priciest.
    categories = {"Home": (0.40, 60), "Books": (0.25, 25), "Toys": (0.20, 45), "Electronics": (0.15, 180)}
    rows = []
    for i, date in enumerate(rng.choice(pd.date_range("2025-01-01", "2025-06-30"), 1000)):
        cat = rng.choice(list(categories), p=[v[0] for v in categories.values()])
        mean = categories[cat][1]
        rows.append({"order_id": f"O{i:04d}", "date": pd.Timestamp(date).strftime("%Y-%m-%d"),
                     "category": cat, "customer_type": rng.choice(["new", "returning"]),
                     "order_value": round(max(5.0, rng.normal(mean, mean * 0.25)), 2)})
    return pd.DataFrame(rows)


def make_deliveries(rng) -> pd.DataFrame:
    base_days = {"FastShip": 3.0, "QuickPost": 4.0, "ParcelPro": 3.5}
    dates = pd.date_range("2025-09-01", "2025-10-31")
    rows = []
    for i in range(1000):
        date = pd.Timestamp(rng.choice(dates))
        carrier = rng.choice(list(base_days))
        delay = 3.0 if (carrier == "FastShip" and date.month == 10) else 0.0
        rows.append({"order_id": f"D{i:04d}", "date": date.strftime("%Y-%m-%d"), "carrier": carrier,
                     "region": rng.choice(["North", "South", "East", "West"]),
                     "delivery_days": round(max(1.0, rng.normal(base_days[carrier] + delay, 0.8)), 1)})
    return pd.DataFrame(rows)


def make_discounts(rng) -> pd.DataFrame:
    rows = []
    for _ in range(800):
        discount = int(rng.integers(0, 41))
        rows.append({"product": rng.choice(["Mug", "Lamp", "Chair", "Rug"]),
                     "discount_pct": discount, "price": round(float(rng.uniform(10, 90)), 2),
                     "units_sold": max(0, int(rng.normal(20 + 0.5 * discount, 4)))})
    return pd.DataFrame(rows)


def make_conversions(rng) -> pd.DataFrame:
    rows = []
    for day, date in enumerate(pd.date_range("2025-01-06", periods=84)):  # weeks 1-12
        sessions = int(rng.normal(2000, 150))
        rows.append({"date": date.strftime("%Y-%m-%d"), "week": day // 7 + 1, "sessions": sessions,
                     "conversions": int(rng.binomial(sessions, 0.03))})  # same 3% rate all along
    return pd.DataFrame(rows)


GENERATORS = {
    "churn.csv": make_churn,
    "orders.csv": make_orders,
    "deliveries.csv": make_deliveries,
    "discounts.csv": make_discounts,
    "conversions.csv": make_conversions,
}


def ensure_datasets() -> None:
    """Write any missing CSV. Each file has its own seed, so files don't depend on each other."""
    DATA_DIR.mkdir(exist_ok=True)
    for seed, (name, make) in enumerate(GENERATORS.items()):
        path = DATA_DIR / name
        if not path.exists():
            make(np.random.default_rng(seed)).to_csv(path, index=False)


if __name__ == "__main__":
    ensure_datasets()
    print(f"Datasets in {DATA_DIR}")
