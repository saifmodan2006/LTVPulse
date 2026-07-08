"""
generate_sample_data.py
------------------------
Creates the three raw files the app expects:
  customer_profiles.json
  order_history.csv
  web_traffic_logs.csv

Deliberately injects messiness (nulls, duplicates, negative amounts,
future-dated orders) so the cleaning pipeline in app.py has real work to do
and the "Anomalous Records Purged" KPI is non-zero.
"""

import json
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

N_CUSTOMERS = 60
REGIONS = ["North", "South", "East", "West"]
PLANS = ["Free", "Basic", "Premium"]
CATEGORIES = ["Electronics", "Apparel", "Home", "Books", "Beauty"]

# ---------------------------------------------------------------
# 1. customer_profiles.json
# ---------------------------------------------------------------
profiles = []
start = datetime(2025, 1, 1)
for i in range(1, N_CUSTOMERS + 1):
    signup = start + timedelta(days=random.randint(0, 400))
    profiles.append({
        "customer_id": f"CUST{i:04d}",
        "name": f"Customer {i}",
        "signup_date": signup.strftime("%Y-%m-%d"),
        "region": random.choice(REGIONS),
        "plan_type": random.choice(PLANS),
    })

# inject a couple of dirty rows: missing id, duplicate row
profiles.append({"customer_id": None, "name": "Ghost User", "signup_date": "2025-02-01",
                  "region": "North", "plan_type": "Free"})
profiles.append(profiles[3])  # exact duplicate

with open("customer_profiles.json", "w") as f:
    json.dump(profiles, f, indent=2)

# ---------------------------------------------------------------
# 2. order_history.csv
# ---------------------------------------------------------------
orders = []
order_id = 1
for p in profiles[:N_CUSTOMERS]:
    signup_dt = datetime.strptime(p["signup_date"], "%Y-%m-%d")
    n_orders = np.random.poisson(4)
    for _ in range(n_orders):
        order_dt = signup_dt + timedelta(days=random.randint(0, 300))
        orders.append({
            "order_id": f"ORD{order_id:05d}",
            "customer_id": p["customer_id"],
            "Purchase_Date": order_dt.strftime("%Y-%m-%d"),
            "Amount_INR": round(np.random.gamma(3, 500), 2),
            "product_category": random.choice(CATEGORIES),
        })
        order_id += 1

# inject anomalies: negative amount, missing customer_id, future date
orders.append({"order_id": f"ORD{order_id:05d}", "customer_id": "CUST0002",
                "Purchase_Date": "2025-03-10", "Amount_INR": -450.00,
                "product_category": "Electronics"})
order_id += 1
orders.append({"order_id": f"ORD{order_id:05d}", "customer_id": None,
                "Purchase_Date": "2025-04-01", "Amount_INR": 999.00,
                "product_category": "Books"})
order_id += 1
orders.append({"order_id": f"ORD{order_id:05d}", "customer_id": "CUST0005",
                "Purchase_Date": "2027-01-01", "Amount_INR": 1200.00,
                "product_category": "Home"})

orders_df = pd.DataFrame(orders)
orders_df.to_csv("order_history.csv", index=False)

# ---------------------------------------------------------------
# 3. web_traffic_logs.csv
# ---------------------------------------------------------------
logs = []
for p in profiles[:N_CUSTOMERS]:
    signup_dt = datetime.strptime(p["signup_date"], "%Y-%m-%d")
    n_sessions = np.random.poisson(8)
    for _ in range(n_sessions):
        session_dt = signup_dt + timedelta(days=random.randint(0, 300))
        logs.append({
            "customer_id": p["customer_id"],
            "session_date": session_dt.strftime("%Y-%m-%d"),
            "page_views": np.random.randint(1, 20),
            "session_duration_sec": np.random.randint(10, 1800),
        })

logs.append({"customer_id": None, "session_date": "2025-05-01",
             "page_views": 3, "session_duration_sec": 120})

traffic_df = pd.DataFrame(logs)
traffic_df.to_csv("web_traffic_logs.csv", index=False)

print("Generated: customer_profiles.json, order_history.csv, web_traffic_logs.csv")
print(f"  profiles: {len(profiles)} rows | orders: {len(orders_df)} rows | traffic: {len(traffic_df)} rows")