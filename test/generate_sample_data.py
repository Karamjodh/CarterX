# generate_sample_data.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

random.seed(42)
np.random.seed(42)

products = [
    ("Laptop",       "Electronics",  800, 1200),
    ("Phone",        "Electronics",  400,  900),
    ("Mouse",        "Accessories",   20,   60),
    ("Keyboard",     "Accessories",   40,  120),
    ("Monitor",      "Electronics",  200,  500),
    ("Desk",         "Furniture",    150,  400),
    ("Chair",        "Furniture",    100,  350),
    ("Headphones",   "Accessories",   50,  200),
    ("Webcam",       "Electronics",   60,  150),
    ("Notebook",     "Stationery",     5,   20),
    ("USB Hub",      "Accessories",   25,   80),
    ("Lamp",         "Furniture",     30,  100),
    ("Phone Case",   "Accessories",   10,   40),
    ("Screen Clean", "Accessories",    5,   20),
    ("Laptop Bag",   "Accessories",   30,  100),
]

# Define realistic product bundles — items commonly bought together
# This ensures association rules will actually be discoverable
BUNDLES = [
    ["Laptop", "Mouse", "Keyboard"],
    ["Laptop", "Mouse", "Laptop Bag"],
    ["Laptop", "Keyboard", "USB Hub"],
    ["Phone", "Phone Case", "Headphones"],
    ["Phone", "Phone Case"],
    ["Monitor", "Mouse", "Keyboard"],
    ["Desk", "Chair", "Lamp"],
    ["Webcam", "USB Hub"],
    ["Mouse", "Keyboard"],
    ["Laptop", "Screen Clean"],
]

rows  = []
txn_id = 1
start_date = datetime(2023, 1, 1)

# Generate 300 customers
for customer_num in range(1, 301):
    customer_id = f"C{customer_num:03d}"

    # Each customer makes 2-8 orders
    n_orders = random.randint(2, 8)

    for _ in range(n_orders):
        date = start_date + timedelta(days=random.randint(0, 364))

        # 60% chance of buying a bundle, 40% chance of single random item
        if random.random() < 0.60:
            bundle = random.choice(BUNDLES)
            items  = bundle
        else:
            items = [random.choice(products)[0]]

        for item_name in items:
            # Find matching product details
            product = next(p for p in products if p[0] == item_name)
            _, category, low, high = product

            rows.append({
                "customer_id":    customer_id,
                "transaction_id": f"T{txn_id:05d}",
                "product_name":   item_name,
                "category":       category,
                "quantity":       random.randint(1, 3),
                "price":          round(random.uniform(low, high), 2),
                "date":           date.strftime("%Y-%m-%d"),
            })

        txn_id += 1

df = pd.DataFrame(rows)
df.to_csv("sample_data.csv", index=False)
print(f"Generated {len(df)} rows")
print(f"Unique customers:   {df['customer_id'].nunique()}")
print(f"Unique transactions:{df['transaction_id'].nunique()}")
print(f"\nSample:\n{df.head(8).to_string()}")