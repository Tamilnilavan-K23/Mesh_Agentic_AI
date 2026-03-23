# =============================================================
#  step7_orchestrator.py
#  Runs both agents for 600 cycles (= 50 hours of telemetry).
#  Handles your CSV column naming (Node_ID, Timestamp, etc.)
#  Prints every ALERT / REROUTE event to console.
#
#  Usage:
#      python step7_orchestrator.py
#      python step7_orchestrator.py --cycles 2000
# =============================================================

import sys
import pandas as pd
import numpy as np
from datetime import datetime

from config import CSV_PATH, MODEL_PATH, COLUMN_RENAME_MAP
from predictive_maintenance_agent import PredictiveMaintenanceAgent
from dynamic_routing_agent import DynamicRoutingAgent
from preprocessor import TelemetryPreprocessor

# ── CLI: optional --cycles argument ────────────────────────
MAX_CYCLES = 600
if "--cycles" in sys.argv:
    try:
        MAX_CYCLES = int(sys.argv[sys.argv.index("--cycles") + 1])
    except (IndexError, ValueError):
        pass

print("=" * 65)
print("  Step 7 — Full Orchestrator")
print("=" * 65)
print(f"  CSV   : {CSV_PATH}")
print(f"  Model : {MODEL_PATH}")
print(f"  Cycles: {MAX_CYCLES}")
print()


# ── Load and normalise the CSV ──────────────────────────────
def load_and_normalise(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Normalise all column names to lowercase snake_case
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("%", "pct")
    )

    # Apply rename map
    rename = {k.lower(): v for k, v in COLUMN_RENAME_MAP.items()}
    df = df.rename(columns=rename)

    # Standardise timestamp column
    ts_col = next((c for c in df.columns if "timestamp" in c or c == "ts"), None)
    if ts_col and ts_col != "timestamp":
        df = df.rename(columns={ts_col: "timestamp"})
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Standardise node_id column
    nid_col = next((c for c in df.columns if "node" in c and "id" in c), None)
    if nid_col and nid_col != "node_id":
        df = df.rename(columns={nid_col: "node_id"})

    print(f"  Loaded {len(df):,} rows | {df['node_id'].nunique()} nodes")
    print(f"  Columns: {list(df.columns)}\n")
    return df


df_run = load_and_normalise(CSV_PATH)

# Group by timestamp → one batch per timestamp tick
batches = [
    grp.to_dict("records")
    for _, grp in df_run.groupby("timestamp")
]
print(f"  Total batches : {len(batches):,}")
print(f"  Running first : {MAX_CYCLES} cycles\n")


# ── Initialise agents ───────────────────────────────────────
pm     = PredictiveMaintenanceAgent(MODEL_PATH)
router = DynamicRoutingAgent()
stats  = []
prev   = None

print("─" * 65)
print("  CYCLE  TIMESTAMP             ALERT   REROUTE  WATCH  OK  QUAR")
print("─" * 65)

for cycle, batch in enumerate(batches[:MAX_CYCLES], 1):

    # ── PM agent cycle ────────────────────────────────────
    decs, summ = pm.run_cycle(batch)
    if prev:
        outcomes = pm.reflect(batch)
        for nid, outcome in outcomes.items():
            if outcome != "recovered":
                print(f"         REFLECT {nid} → {outcome}")

    # ── Routing agent ─────────────────────────────────────
    router.update_topology(batch, pm.quarantined_nodes)
    router.compute_all_routes("HQ")

    # ── Collect stats ─────────────────────────────────────
    ts = batch[0].get("timestamp", cycle)
    stats.append({
        "cycle":      cycle,
        "ts":         ts,
        "alert":      len(summ["alert"]),
        "reroute":    len(summ["reroute"]),
        "watch":      len(summ["watch"]),
        "ok":         len(summ["ok"]),
        "quarantined":len(pm.quarantined_nodes),
    })

    # ── Print only non-trivial cycles ────────────────────
    if summ["alert"] or summ["reroute"] or summ["watch"]:
        a = summ["alert"]
        r = summ["reroute"]
        w = summ["watch"]
        o = summ["ok"]
        q = list(pm.quarantined_nodes)
        print(f"  {cycle:05d}  {str(ts)[:19]:<21}  "
              f"A={a}  R={r}  W={w}  OK={o}  Q={q}")

    prev = batch

# ── Final report ────────────────────────────────────────────
stats_df = pd.DataFrame(stats)
print("\n" + "=" * 65)
print("  FINAL REPORT")
print("=" * 65)
print(f"  Cycles run      : {len(stats_df):,}")
print(f"  Total ALERT     : {stats_df['alert'].sum()}")
print(f"  Total REROUTE   : {stats_df['reroute'].sum()}")
print(f"  Total WATCH     : {stats_df['watch'].sum()}")
print(f"  Peak quarantined: {stats_df['quarantined'].max()} nodes at once")

from collections import Counter
q_counts = Counter()
for d in pm.decision_log:
    if d.action in ("alert", "reroute"):
        q_counts[d.node_id] += 1
if q_counts:
    print("\n  Most flagged nodes:")
    for nid, cnt in q_counts.most_common(5):
        print(f"    {nid}  →  {cnt} flag events")

# Save stats for visualisation
stats_df.to_csv("orchestrator_stats.csv", index=False)
print("\n  Stats saved to orchestrator_stats.csv")
print("  Run step8_visualize.py to generate charts.")
print("=" * 65)

# Make stats_df and pm available if imported as a module
__all__ = ["stats_df", "pm", "router"]
