# Disaster Mesh — Agentic AI (Local Setup)
=========================================

## Folder structure

```
disaster_mesh/
├── config.py                      ← Edit your file paths here
├── preprocessor.py                ← Telemetry validation + normalization
├── predictive_maintenance_agent.py← PM Agent (6-stage loop)
├── dynamic_routing_agent.py       ← Routing Agent (Dijkstra)
├── step5_pm_agent_demo.py         ← Test PM agent alone
├── step6_routing_demo.py          ← Test routing agent alone
├── step7_orchestrator.py          ← Full 600-cycle run
├── step8_visualize.py             ← Generate charts
├── main.py                        ← Run everything at once
├── requirements.txt
├── data/
│   └── validation_dataset.csv     ← your telemetry CSV
└── models/
    └── random_forest_model_fixed.pkl  ← your trained model
```

---

## Setup (one time only)

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate          # Mac/Linux
venv\Scripts\activate             # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create the data and models folders
mkdir data models

# 4. Copy your files into the folders:
#    data/validation_dataset.csv
#    models/random_forest_model_fixed.pkl

# 5. Open config.py and confirm the paths match:
#    CSV_PATH   = "data/validation_dataset.csv"
#    MODEL_PATH = "models/random_forest_model_fixed.pkl"
```

---

## Running the scripts

### Run everything at once (recommended)
```bash
python main.py
python main.py --cycles 2000       # see all failure episodes
python main.py --step 5            # run only one step
```

### Run each step separately
```bash
python step5_pm_agent_demo.py      # verify PM agent predictions
python step6_routing_demo.py       # verify routing around failed node
python step7_orchestrator.py       # full 600-cycle replay
python step7_orchestrator.py --cycles 2000
python step8_visualize.py          # generate charts (run after step 7)
```

---

## What each file does

| File | Purpose |
|------|---------|
| `config.py` | All thresholds, feature names, file paths — edit once |
| `preprocessor.py` | Validates packets, normalizes features, detects breaches |
| `predictive_maintenance_agent.py` | Observe→Perceive→Reason→Plan→Act→Reflect loop |
| `dynamic_routing_agent.py` | Dijkstra mesh routing, avoids quarantined nodes |
| `step5_pm_agent_demo.py` | Healthy vs failing scenario tests |
| `step6_routing_demo.py` | Shows routing table with N03 quarantined |
| `step7_orchestrator.py` | Replays full CSV, prints every ALERT/REROUTE event |
| `step8_visualize.py` | Generates agent_results.png + cumulative_events.png |
| `main.py` | Runs all steps in sequence |

---

## CSV column name support

The system automatically handles these column name variations:

| Your CSV column | What the agent uses |
|-----------------|---------------------|
| `packet_loss_pct` | `pkt_loss_pct` |
| `pkt_loss_%` | `pkt_loss_pct` |
| `queue_length` | `queue_len_pkts` |
| `Node_ID` | `node_id` |
| `Timestamp` | `timestamp` |
| `Label` | `label` |

---

## Outputs

After running step 7 + step 8:
- `orchestrator_stats.csv` — cycle-by-cycle action counts
- `agent_results.png` — timeline chart (alerts/reroutes/healthy)
- `cumulative_events.png` — running total of events
