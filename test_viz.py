# test_viz.py

import base64
import numpy as np
from sleepqualityvisualization import SleepQualityVisualizer

# ─── Mock Sleep Data ────────────────────────────────────────────────────────────

# Simulate an 8-hour sleep with 1 sample every 30s → 8*3600/30 = 960 points
n_samples = 960
times = np.arange(n_samples)

sleep_data = {
    "start_time": "2025-05-05T22:00:00",
    "end_time":   "2025-05-06T06:00:00",
    "sleep_stages": {"deep": 90, "rem": 80, "light": 200, "awake": 10},
    "heart_rate": (60 + 5 * np.sin(2 * np.pi * times / n_samples)).tolist(),
    "hrv":        (50 + 10 * np.cos(2 * np.pi * times / n_samples)).tolist(),
    "spo2":       (98 + 0.5 * np.sin(4 * np.pi * times / n_samples)).tolist(),
    "wrist_temp": (36.5 + 0.2 * np.sin(3 * np.pi * times / n_samples)).tolist(),
    "date":       "2025-05-06"
}

# A small history for trend plotting
trend_data = [
    {"date": "2025-05-01", "sleep_quality_score": 75},
    {"date": "2025-05-02", "sleep_quality_score": 80},
    {"date": "2025-05-03", "sleep_quality_score": 78},
    {"date": "2025-05-04", "sleep_quality_score": 82},
    {"date": "2025-05-05", "sleep_quality_score": 79},
    {"date": "2025-05-06", "sleep_quality_score": 83},
]

sleep_score = 83  # example final score

# ─── Generate & Save Charts ────────────────────────────────────────────────────

viz = SleepQualityVisualizer(theme="light")

def save_png(b64_str: str, fname: str):
    with open(fname, "wb") as f:
        f.write(base64.b64decode(b64_str))

# 1) Score card
save_png(viz.generate_score_card(sleep_score), "score_card.png")

# 2) Trend over days
save_png(viz.generate_sleep_quality_trend(trend_data), "trend.png")

# 3) Sleep structure pie
save_png(viz.generate_sleep_structure_chart(sleep_data), "structure.png")

# 4) Heart rate
save_png(viz.generate_heart_rate_chart(sleep_data), "heart_rate.png")

# 5) HRV
save_png(viz.generate_hrv_trend(sleep_data), "hrv.png")

# 6) SpO₂
save_png(viz.generate_spo2_trend(sleep_data), "spo2.png")

# 7) Temperature
save_png(viz.generate_temperature_trend(sleep_data), "temperature.png")

print("✓ All 7 charts written to current folder.")
