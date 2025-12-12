import numpy as np
import pandas as pd

np.random.seed(42)
N = 10000

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

rows = []

for _ in range(N):
    age = np.random.randint(18, 75)
    tenure_days = clamp(np.random.exponential(400), 0, 3650)  # cap ~10 years
    rentals = clamp(int(tenure_days / 120), 0, 60)
    rentals = int(tenure_days / 120)
    chargebacks = np.random.choice([0,1,2], p=[0.92,0.06,0.02])

    amount = np.random.lognormal(mean=5.3, sigma=0.6)
    amount = clamp(amount, 30, 3500)


    duration = np.random.randint(1, 14)
    lead_time = np.random.exponential(24)
    distance = np.random.exponential(50)
    distance = clamp(distance, 0, 800)

    failed_attempts = np.random.choice([0,1,2,3,5], p=[0.7,0.15,0.08,0.05,0.02])

    booking = np.random.choice(["web","app","phone","walk_in"], p=[0.45,0.35,0.1,0.1])
    card_present = np.random.choice(["yes","no"], p=[0.75,0.25])
    same_name = np.random.choice(["yes","no"], p=[0.9,0.1])
    billing_match = np.random.choice(["yes","no"], p=[0.92,0.08])
    ip_match = np.random.choice(["yes","no"], p=[0.9,0.1])

    # Risk score (latent)
    risk = 0.0
    risk += 0.20 if chargebacks >= 1 else 0

    risk += 0.15 if lead_time < 4 else 0
    risk += 0.15 if card_present == "no" else 0
    risk += 0.2 if failed_attempts >= 3 else 0
    risk += 0.1 if billing_match == "no" else 0
    risk -= 0.1 if tenure_days > 365 else 0
    risk -= 0.05 if rentals >= 5 else 0

    fraud_prob = clamp(0.05 + risk + np.random.normal(0,0.05), 0, 1)
    is_fraud = np.random.rand() < fraud_prob

    rows.append({
        "rental_amount": round(amount,2),
        "rental_duration_days": duration,
        "lead_time_hours": round(lead_time,1),
        "distance_from_home_to_branch_km": round(distance,1),
        "customer_age": age,
        "customer_tenure_days": int(tenure_days),
        "previous_rentals_count": rentals,
        "previous_chargebacks_count": chargebacks,
        "failed_attempts_last_24h": failed_attempts,
        "booking_channel": booking,
        "card_present": card_present,
        "same_name_on_card": same_name,
        "billing_matches_id": billing_match,
        "ip_country_matches_id_country": ip_match,
        "is_fraud": int(is_fraud)
    })

df = pd.DataFrame(rows)
df.to_csv("soltrice_fraud_data.csv", index=False)

print("Generated soltrice_fraud_data.csv with", len(df), "rows")

