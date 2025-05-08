# test_analysis.py

import base64
import numpy as np
from sleep_quality_analysis import (
    generate_hr_chart,
    generate_hrv_chart,
    generate_spo2_chart,
    generate_temp_chart,
    generate_sleep_stage_pie_chart,
    build_sleep_cnn,
    compute_channel_importance
)

# ─── Helper to write PNG from base64 ────────────────────────────────────────────
def save_png(b64_str: str, filename: str):
    with open(filename, "wb") as f:
        f.write(base64.b64decode(b64_str))

# ─── Mock Sleep Data for One Night ─────────────────────────────────────────────
n = 480  # e.g. 8 hours × 60min ÷ (1 sample/minute)
times = np.arange(n)

sleep_data = {
    "start_time": "2025-05-05T22:00:00",
    "end_time":   "2025-05-06T06:00:00",
    "sleep_stages": {"deep": 90, "rem": 80, "light": 200, "awake": 10},
    "heart_rate":   (60 + 5 * np.sin(2 * np.pi * times / n)).tolist(),
    "hrv":          (50 + 8 * np.cos(2 * np.pi * times / n)).tolist(),
    "spo2":         (98 + 0.5 * np.sin(4 * np.pi * times / n)).tolist(),
    "wrist_temp":   (36.5 + 0.2 * np.sin(3 * np.pi * times / n)).tolist(),
    "date":         "2025-05-06"
}

# Example final score
score = 85

# ─── Generate & Save Individual Charts ─────────────────────────────────────────
save_png(generate_hr_chart(sleep_data),                "hr.png")
save_png(generate_hrv_chart(sleep_data),               "hrv.png")
save_png(generate_spo2_chart(sleep_data),              "spo2.png")
save_png(generate_temp_chart(sleep_data),              "temp.png")
save_png(generate_sleep_stage_pie_chart(sleep_data),   "stages.png")

print("✓ Charts generated: hr.png, hrv.png, spo2.png, temp.png, stages.png")

# ─── Build & Train Demo CNN Model ──────────────────────────────────────────────
model = build_sleep_cnn(input_length=n, channels=4)
model.summary()

# Fake training data for demonstration
X = np.random.randn(10, n, 4).astype(np.float32)
y = np.random.randint(50, 100, size=(10, 1)).astype(np.float32)

model.fit(X, y, epochs=2, batch_size=2, verbose=1)

# Compute and display channel importances
importances = compute_channel_importance(model)
print("Channel importances:", importances)
