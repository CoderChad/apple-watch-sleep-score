# Apple Watch Sleep Scoring System

A comprehensive AI/ML pipeline that ingests Apple Watch biometric data, computes a scientifically grounded sleep quality score (0–100), and produces actionable insights and visualizations. Designed as a Python prototype, it can be extended into an iOS app using HealthKit, Core ML, and SwiftUI.

---

## 🚀 Intent

* Offer users an **objective measure** of nightly sleep quality by combining heart rate, HRV, blood oxygen, temperature, and sleep-stage durations into a single score.
* Provide **interpretability** through per‐metric visualizations and stage breakdowns.
* Enable **tracking over time** so individuals can identify trends, routines, and areas for improvement.
* Lay groundwork for **seamless integration** into the Apple ecosystem (HealthKit, Core ML, SwiftUI).

---

## 📊 Metrics Used

| Metric                | Source              | Role                                       |
| --------------------- | ------------------- | ------------------------------------------ |
| **Heart Rate (BPM)**  | Apple Watch HR      | Recovery pattern analysis                  |
| **HRV**               | Watch HRV API       | Autonomic balance / sleep depth indicator  |
| **SpO₂**              | Blood oxygen sensor | Identify desaturation events               |
| **Wrist Temperature** | Watch temperature   | Circadian fluctuation and deep‐sleep proxy |
| **Sleep Stages**      | HealthKit stages    | Deep / REM / light / awake proportions     |

---

## 🏆 Expected Outcomes

1. **Sleep Quality Score (0–100)** – combines multiple features into a single, intuitive metric.
2. **User Insights** – tailored tips (e.g., improve continuity, optimize routine) based on feature values.
3. **Visual Analytics** – per-night charts (time series + pie charts) and trend graphs for longitudinal monitoring.

---

## 📁 File Responsibilities

```text
apple-watch-sleep-score/
├── src/
│   ├── data_ingestion.py       # Parse HealthKit / WatchKit exports into DataFrames
│   ├── preprocessing.py        # Resampling, missing‐value handling, stage‐duration extraction
│   ├── feature_engineering.py  # Compute heart‐rate patterns, stage‐balance, normalization
│   ├── model/
│   │   ├── train.py            # Train & persist sleep‐score model (RandomForest / CNN)
│   │   └── predict.py          # Load model & predict on new data
│   ├── visualization.py        # Generate time‐series & pie‐chart PNGs (Base64) for each metric
│   └── utils.py                # Logging, I/O helpers, config paths
├── tests/                      # pytest unit tests for each module
├── notebooks/                  # Exploratory data analysis & prototyping (Jupyter)
├── environment.yml             # Conda environment specification
├── requirements.txt            # pip requirements for reproducibility
├── .github/workflows/ci.yml    # CI pipeline: linting, testing on push & PR
├── .gitignore                  # Ignore data exports, virtual environments, caches
├── README.md                   # Project overview, setup & usage instructions
└── LICENSE                     # MIT License
```

---

## 🍏 Apple Ecosystem Integration

1. **HealthKit Ingestion** – Replace Python XML parsing with Swift’s HealthKit APIs to fetch HR, HRV, SpO₂, temperature, and sleep-stage samples directly on device.
2. **Core ML Model** – Convert trained Python model (RandomForest or CNN) to `.mlmodel` format via `coremltools`, enabling on‐device prediction:

   ```bash
   coremltools convert sleep_score_model.joblib \
     --output sleep_score.mlmodel \
     --class-labels heart_rate_pattern hr…
   ```
3. **SwiftUI Dashboards** – Render time‐series and pie charts using SwiftUI’s `Chart` framework, loading PNGs or re‐implementing in‐native using `Path` & `Shape`.
4. **WidgetKit & Notifications** – Expose nightly scores in Widgets and schedule notifications for unusual sleep patterns.
5. **Watch App Companion** – Host minimal logic on watchOS for real‐time HR/HRV monitoring and push data to iPhone app for scoring.

---

## 📖 Next Steps

* **Clone & Setup**:

  ```bash
  git clone git@github.com:CoderChad/apple-watch-sleep-score.git
  cd apple-watch-sleep-score
  conda env create -f environment.yml && conda activate sleep-score
  pytest  # ensure tests pass
  ```

* **Train & Evaluate**:

  ```bash
  python src/model/train.py --data data/processed/features.npy --labels data/processed/labels.npy --out model/sleep_score.joblib
  ```

* **Visualize**:

  ```python
  from src.visualization import plot_sleep_metric
  # … load DataFrame, call functions, save PNGs …
  ```

* **Publish**: Tag releases via semantic versioning and update `CHANGELOG.md` accordingly.

---

*Distributed under the MIT License.*
