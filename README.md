# 🤖 Robot Predictive Maintenance — Neon + Live Streaming Alerts + Dashboard

**Executive intent:** turn raw robot telemetry into **actionable predictive-maintenance signals** using  
(1) **model training + learned thresholds**, (2) **live streaming detection**, and (3) a **dashboard** — all persisted in **Neon (Postgres)**.

---

## 🧠 Problem

Industrial robots produce continuous sensor readings (current/axis signals). Failures are expensive and often detected late.  
This project watches streaming signals and raises early warnings:

- ⚠️ **ALERT** → abnormal behavior emerging (schedule maintenance soon)
- 🛑 **ERROR** → high-risk abnormality (urgent intervention)

---

## ✅ What You Can Demo (Deliverables)

### 1) Notebook: Training + Threshold Learning (Neon)
File: `notebooks/01_train_models_thresholds_neon.ipynb`

What it does:
- Load historical robot dataset
- Fit a **Linear Regression baseline** (per robot / per signal)
- Compute residual-based thresholds:
  - `residual_alert` (early warning)
  - `residual_error` (critical)
- Save trained params + thresholds into Neon: `linear_regression.models`

### 2) Notebook: Live Streaming + Alerts + Event Logging (Neon)
File: `notebooks/02_streaming_alerts_dashboard_neon.ipynb`

What it does:
- Stream recent points per robot (smooth + readable)
- Plot per-robot panels:
  - observed signal
  - smoothed signal
  - regression baseline
  - threshold bands
  - ⚠️ / 🛑 markers when events trigger
- Save events to:
  - `experiments/events.log` (local audit log)
  - `linear_regression.events` (Neon table)

### 3) Dashboard (Streamlit)
File: `dashboard/app.py`

What it does:
- Pull latest stream + events from Neon
- Show per-robot “operator view”
- Summarize events over a lookback window

---

## 🧱 Architecture (High-Level)

**Raw CSV → training pipeline → models in Neon → streaming detector → events in Neon → dashboard**

- **Data layer:** CSV + Neon Postgres (persistent, dashboard-ready)
- **Model layer:** linear regression baseline (interpretable, fast)
- **Detection layer:** residual thresholding + cooldown to reduce alert spam
- **Observability:** events.log + Neon `events` table + dashboard panels

---

## 📁 Project Structure

```text
.
├─ configs/
├─ dashboard/
│  └─ app.py
├─ data/
│  ├─ processed/
│  │  └─ processed_robot_data.csv
│  └─ raw/
│     └─ RMBR4-2_export_test_with_robotids_*.csv
├─ experiments/
│  ├─ plots/
│  │  ├─ robot_1_live.html
│  │  ├─ robot_2_live.html
│  │  ├─ robot_3_live.html
│  │  └─ robot_4_live.html
│  ├─ events_log.csv
│  ├─ events_robot_1.csv
│  ├─ events_robot_2.csv
│  ├─ events_robot_3.csv
│  ├─ events_robot_4.csv
│  ├─ events.log
│  └─ results.csv
├─ notebooks/
│  ├─ 01_train_models_thresholds_neon.ipynb
│  ├─ 02_streaming_alerts_dashboard_neon.ipynb
│  └─ Optional-Notebook(practise).ipynb
├─ screenshots/
├─ venv/                 # local virtual environment (don’t commit)
├─ .flake8
├─ .gitignore
├─ README.md
└─ requirements.txt
```

---

## ⚙️ Setup

### 1) Create & activate a virtual environment
```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\activate
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) Configure Neon connection
Create a `.env` file in the project root:

```env
PGHOST=xxxxx.neon.tech
PGDATABASE=xxxx
PGUSER=xxxx
PGPASSWORD=xxxx
PGPORT=5432
PGSSLMODE=require
```

---

## ▶️ How to Run (Correct Order)

### Step 1 — Train models + save thresholds to Neon
Run:
- `notebooks/01_train_models_thresholds_neon.ipynb`

Expected:
- ✅ `linear_regression.models` populated (typically 4 robots / 4 rows)
- Threshold values visible in output tables/prints

### Step 2 — Run streaming + generate events + save logs
Run:
- `notebooks/02_streaming_alerts_dashboard_neon.ipynb`

Expected:
- 4 robot plots with baseline + threshold bands
- ⚠️ and 🛑 markers appear when residual exceeds thresholds
- `experiments/events.log` is written/updated
- ✅ `linear_regression.events` populates as streaming runs

### Step 3 — Launch the Streamlit dashboard

After you finish **Notebook 2** (it generates the latest streaming events/logs), start the dashboard from your project root:

**Windows (PowerShell / CMD):**
```bash
streamlit run dashboard\\app.py
```

**macOS / Linux:**
```bash
streamlit run dashboard/app.py
```

Expected:
- Robot panels + recent events summary pulled from Neon
- A local URL like `http://localhost:8501`


---

## 🗃️ Database Tables (Neon)

Common tables used/created:
- `linear_regression.models` → model coefficients + thresholds
- `linear_regression.events` → alert/error events with timestamps
- (optional in your implementation) `training_points`, `stream_points`

---

## 📌 Key Design Choices (and how to explain them)

### Why a Linear Regression baseline?
- Interpretable and fast → perfect baseline for workshop-grade predictive maintenance
- Easier to validate than complex models (clear “expected vs observed”)

### Why residual-based thresholds?
- Residual = **observed − expected**
- Converts continuous deviation into actionable categories:
  - `residual_alert` = early anomaly
  - `residual_error` = critical anomaly

### Why cooldown logic?
- Streaming detectors can spam repeated alerts
- Cooldown improves signal-to-noise and creates a cleaner operator experience

### Why Neon DB?
- Production-like persistence (not just notebook memory)
- Enables dashboard queries and reproducible demos

---

### 30-second overview
- “This project turns robot sensor streams into predictive-maintenance alerts.”
- “We learn thresholds from historical data, then apply them on streaming data in real time.”
- “All models and events persist to Neon, and Streamlit shows an operator dashboard.”

### Notebook 1 (Training + thresholds)
- “This notebook fits a baseline model to learn what *normal* looks like.”
- “Residual thresholds become our ALERT/ERROR rules and are stored in `linear_regression.models`.”

### Notebook 2 (Streaming + alerts)
- “Here we stream points and compare observed vs expected in real time.”
- “When residual crosses the learned thresholds, we log ⚠️/🛑 events to both a file and Neon.”

### Dashboard
- “This is the operator view: it pulls the latest signals and events directly from Neon.”
- “The key idea is end-to-end reproducibility: data → model → events → dashboard.”

---

## ✅ Submission Checklist

- [ ] Notebooks run end-to-end on a fresh machine after `pip install -r requirements.txt`
- [ ] `.env` is present locally and **NOT committed**
- [ ] `linear_regression.models` has rows after Notebook 1
- [ ] `linear_regression.events` populates after Notebook 2 streaming
- [ ] Dashboard launches and shows recent events

---

## Author

**Param Avinashkumar Rasaniya**  
Course: Predictive Maintenance / Streaming Analytics Workshop
