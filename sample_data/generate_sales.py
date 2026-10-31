"""Generate sample_data/sales.csv (seeded, so the output is reproducible).

Run from the project root:  python sample_data/generate_sales.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROWS = 1500
rng = np.random.default_rng(42)

products = {  # name: (base price, weight)
    "Laptop": (900, 0.15),
    "Monitor": (300, 0.20),
    "Keyboard": (60, 0.25),
    "Mouse": (30, 0.25),
    "Headset": (120, 0.15),
}
regions = {"North": 0.25, "South": 0.20, "East": 0.25, "West": 0.30}
salespeople = ["Asha", "Ben", "Carla", "Dev", "Elena", "Farid"]

names = list(products)
product = rng.choice(names, ROWS, p=[products[n][1] for n in names])
region = rng.choice(list(regions), ROWS, p=list(regions.values()))
salesperson = rng.choice(salespeople, ROWS)
quantity = rng.integers(1, 11, ROWS)
unit_price = np.array([products[p][0] for p in product]) * rng.uniform(0.9, 1.1, ROWS)
revenue = np.round(unit_price * quantity, 2)
date = pd.to_datetime("2023-01-01") + pd.to_timedelta(rng.integers(0, 731, ROWS), unit="D")

df = pd.DataFrame(
    {
        "date": date.strftime("%Y-%m-%d"),
        "product": product,
        "region": region,
        "revenue": revenue,
        "quantity": quantity,
        "salesperson": salesperson,
    }
).sort_values("date", kind="stable")

# a few missing values so the null summary is non-trivial
df = df.reset_index(drop=True)
df.loc[rng.choice(ROWS, 12, replace=False), "salesperson"] = np.nan
df.loc[rng.choice(ROWS, 8, replace=False), "quantity"] = np.nan

out = Path(__file__).resolve().parent / "sales.csv"
df.to_csv(out, index=False)
print(f"Wrote {len(df)} rows to {out}")
