# anomaly_detection.py
"""
Argus-AI Prototype: Early Detection of Antibiotic Shortages
This script simulates hospital antibiotic usage, detects anomalies using Isolation Forest,
and visualizes emerging shortage alerts.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# -----------------------------
# Step 1: Simulate Sample Data
# -----------------------------
dates = pd.date_range(start='2026-01-01', periods=30)
hospitals = ['HOSP001', 'HOSP002']
antibiotics = ['ceftriaxone', 'ciprofloxacin']

records = []
np.random.seed(42)
for date in dates:
    for hosp in hospitals:
        # Normal usage
        cef = np.random.randint(80, 130)
        cipro = np.random.randint(30, 60)
        # Introduce anomaly: sudden drop in ceftriaxone on day 20
        if date == pd.Timestamp('2026-01-20') and hosp == 'HOSP001':
            cef = 20
        records.append([date, hosp, 'ceftriaxone', cef])
        records.append([date, hosp, 'ciprofloxacin', cipro])

data = pd.DataFrame(records, columns=['date', 'hospital_id', 'antibiotic', 'doses_prescribed'])

# -----------------------------
# Step 2: Pivot Data for Modeling
# -----------------------------
pivot_data = data.pivot_table(
    index='date',
    columns=['hospital_id', 'antibiotic'],
    values='doses_prescribed',
    fill_value=0
)

# -----------------------------
# Step 3: Anomaly Detection
# -----------------------------
model = IsolationForest(contamination=0.05, random_state=42)
pivot_data['anomaly'] = model.fit_predict(pivot_data)

# Extract anomalies
alerts = pivot_data[pivot_data['anomaly'] == -1]
print("Emerging shortage alerts:")
print(alerts)

# -----------------------------
# Step 4: Visualization
# -----------------------------
pivot_data['total_usage'] = pivot_data.sum(axis=1)

plt.figure(figsize=(12,6))
plt.plot(pivot_data.index, pivot_data['total_usage'], label='Total Antibiotic Usage')
plt.scatter(alerts.index, alerts['total_usage'], color='red', s=100, label='Anomaly Alert')
plt.title("Argus-AI: Emerging Antibiotic Shortages")
plt.xlabel("Date")
plt.ylabel("Doses Prescribed")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
