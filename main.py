#!/usr/bin/env python3
# =============================================================
#  main.py — Run all steps in sequence
#  This is the single entry point to run the full system.
#
#  Usage:
#      python main.py                  # all steps
#      python main.py --cycles 2000    # change cycle count
#      python main.py --step 5         # run only one step
# =============================================================

import os
import sys
import subprocess
import pickle
import joblib
import warnings
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict, deque
import matplotlib
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ── Parse CLI args ───────────────────────────────────────────
MAX_CYCLES = 600
ONLY_STEP  = None

for i, arg in enumerate(sys.argv[1:], 1):
    if arg == "--cycles" and i < len(sys.argv):
        try: MAX_CYCLES = int(sys.argv[i + 1])
        except: pass
    if arg == "--step" and i < len(sys.argv):
        try: ONLY_STEP = int(sys.argv[i + 1])
        except: pass

# ── Imports (all local) ─────────────────────────────────────
from config import CSV_PATH, MODEL_PATH, COLUMN_RENAME_MAP, FEATURES
from preprocessor import TelemetryPreprocessor
from predictive_maintenance_agent import PredictiveMaintenanceAgent
from dynamic_routing_agent import DynamicRoutingAgent
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, ConfusionMatrixDisplay


def banner(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# ============================================================
#  STEP 1 — Verify environment
# ============================================================
if not ONLY_STEP or ONLY_STEP == 1:
    banner("Step 1 — Environment check")
    import sklearn, numpy, pandas
    print(f"  scikit-learn : {sklearn.__version__}")
    print(f"  numpy        : {numpy.__version__}")
    print(f"  pandas       : {pandas.__version__}")
    print(f"  Python       : {sys.version.split()[0]}")
    print("  All imports OK")


# ============================================================
#  STEP 2 — Verify file paths
# ============================================================
if not ONLY_STEP or ONLY_STEP == 2:
    banner("Step 2 — File paths")
    print(f"  CSV_PATH   = {CSV_PATH}")
    print(f"  MODEL_PATH = {MODEL_PATH}")

    csv_ok   = os.path.exists(CSV_PATH)
    model_ok = os.path.exists(MODEL_PATH)

    print(f"  CSV found  : {'YES' if csv_ok   else 'NO — will auto-generate'}")
    print(f"  Model found: {'YES' if model_ok else 'NO — will train in Step 4'}")


# ============================================================
#  STEP 3 — Load / generate dataset
# ============================================================
if not ONLY_STEP or ONLY_STEP == 3:
    banner("Step 3 — Load dataset")

    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        # Normalise columns
        df.columns = (df.columns.str.strip().str.lower()
                      .str.replace(" ", "_").str.replace("%", "pct"))
        rename = {k.lower(): v for k, v in COLUMN_RENAME_MAP.items()}
        df = df.rename(columns=rename)
        for col in ["node_id","label","timestamp"]:
            if col not in df.columns:
                candidates = [c for c in df.columns if col.replace("_","") in c.replace("_","")]
                if candidates:
                    df = df.rename(columns={candidates[0]: col})
        print(f"  Loaded   : {len(df):,} rows | {df['node_id'].nunique()} nodes")
        if "label" in df.columns:
            print(f"  Label 0  : {(df.label==0).sum():,} healthy")
            print(f"  Label 1  : {(df.label==1).sum():,} failure")
    else:
        print("  CSV not found — generating synthetic 21-day dataset...")
        import random; random.seed(42); np.random.seed(42)
        NODES = [f"N{i:02d}" for i in range(1, 11)]
        START = datetime(2024, 3, 1)
        STEPS = 21 * 24 * 12
        sched = {
            "N01":[(1200,80),(4500,60),(7800,100)],
            "N02":[(800,50),(3200,70),(6100,90),(8500,55)],
            "N03":[(2100,65),(5500,80),(9000,75)],
            "N04":[(600,45),(4000,55),(7200,85)],
            "N05":[(1500,90),(3800,60),(6800,70),(9200,50)],
            "N06":[(950,55),(4200,80),(8000,95)],
            "N07":[(1800,70),(5000,65),(7500,85)],
            "N08":[(700,50),(3500,75),(6500,60),(8800,90)],
            "N09":[(1100,60),(4800,85),(7100,70),(9500,45)],
            "N10":[(550,40),(3000,65),(6200,95),(8200,80)],
        }
        def fs(step, sc, pre=30):
            for s, d in sc:
                if s-pre<=step<s: return 1, (step-(s-pre))/pre
                if s<=step<s+d:   return 1, 1.0
                if s+d<=step<s+d+10: return 0, 0.3
            return 0, 0.0
        def nz(b, s, lo, hi):
            return float(np.clip(np.random.normal(b, s), lo, hi))
        rows = []
        for node in NODES:
            up = 0.0
            for step in range(STEPS):
                ts  = START + timedelta(minutes=step * 5)
                lbl, deg = fs(step, sched[node])
                if lbl == 1 and deg == 1.0:
                    up = max(0, up - np.random.uniform(0, 0.5))
                else:
                    up += 5 / 60
                rows.append({
                    "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "node_id": node,
                    "battery_v":      round(nz(3.85-deg*.55, .05+deg*.08, 2.9, 4.2), 3),
                    "rssi_dbm":       round(nz(-65-deg*38,   4+deg*8,   -120, -40),  1),
                    "pkt_loss_pct":   round(nz(1.5+deg*30,   .5+deg*6,    0,  100),  2),
                    "temp_c":         round(nz(38+deg*28,    2+deg*4,    20,   85),   1),
                    "uptime_hrs":     round(up, 2),
                    "hop_count":      int(np.clip(np.random.poisson(2+deg*4),  1, 10)),
                    "queue_len_pkts": int(np.clip(np.random.poisson(3+deg*18), 0, 50)),
                    "tx_rate_kbps":   round(nz(85-deg*75, 5+deg*12, 1, 250), 1),
                    "label": lbl,
                })
        df = pd.DataFrame(rows)
        os.makedirs(os.path.dirname(CSV_PATH) if os.path.dirname(CSV_PATH) else ".", exist_ok=True)
        df.to_csv(CSV_PATH, index=False)
        print(f"  Generated : {len(df):,} rows → saved to {CSV_PATH}")


# ============================================================
#  STEP 4 — Train model (skipped if model already exists)
# ============================================================
if not ONLY_STEP or ONLY_STEP == 4:
    banner("Step 4 — Train / load model")
    if os.path.exists(MODEL_PATH):
        bundle  = joblib.load(MODEL_PATH)
        model   = bundle["model"]
        scaler  = bundle["scaler"]
        print(f"  Model loaded: n_estimators={model.n_estimators}")
        print(f"  Features    : {bundle['features']}")
        print("  Step 4 skipped — model already exists.")
    else:
        if "df" not in dir():
            df = pd.read_csv(CSV_PATH)
        feat_cols = [f for f in FEATURES if f in df.columns]
        X = df[feat_cols]; y = df["label"]
        X_tr,X_te,y_tr,y_te = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_te_s = sc.transform(X_te)
        mdl = RandomForestClassifier(
            n_estimators=100, class_weight="balanced",
            random_state=42, n_jobs=-1)
        mdl.fit(X_tr_s, y_tr)
        print(f"  Train: {mdl.score(X_tr_s,y_tr)*100:.1f}%  "
              f"Test: {mdl.score(X_te_s,y_te)*100:.1f}%")
        print(classification_report(y_te, mdl.predict(X_te_s),
                                    target_names=["Healthy","Failure"]))
        os.makedirs(os.path.dirname(MODEL_PATH) if os.path.dirname(MODEL_PATH) else ".", exist_ok=True)
        joblib.dump({"model":mdl,"scaler":sc,"features":feat_cols}, MODEL_PATH)
        print(f"  Saved to: {MODEL_PATH}")


# ============================================================
#  STEP 5 — PM Agent demo
# ============================================================
if not ONLY_STEP or ONLY_STEP == 5:
    banner("Step 5 — PM Agent demo")
    pm = PredictiveMaintenanceAgent(MODEL_PATH)

    healthy = [
        {"node_id":"N01","battery_v":3.9,"rssi_dbm":-65,"pkt_loss_pct":1.2,
         "temp_c":38,"uptime_hrs":48,"hop_count":2,"queue_len_pkts":3,"tx_rate_kbps":88},
        {"node_id":"N02","battery_v":3.8,"rssi_dbm":-70,"pkt_loss_pct":2.1,
         "temp_c":41,"uptime_hrs":72,"hop_count":3,"queue_len_pkts":5,"tx_rate_kbps":82},
    ]
    print("\n  Scenario A — Healthy nodes:")
    decs, summ = pm.run_cycle(healthy)
    for d in decs:
        print(f"    {d.node_id}: {d.action.upper():<8}  {d.reason}")

    failing = [
        {"node_id":"N03","battery_v":3.1,"rssi_dbm":-102,"pkt_loss_pct":38,
         "temp_c":69,"uptime_hrs":0.1,"hop_count":8,"queue_len_pkts":25,"tx_rate_kbps":7},
        {"node_id":"N04","battery_v":3.85,"rssi_dbm":-67,"pkt_loss_pct":1.5,
         "temp_c":39,"uptime_hrs":96,"hop_count":2,"queue_len_pkts":4,"tx_rate_kbps":87},
    ]
    print("\n  Scenario B — N03 critical:")
    decs, summ = pm.run_cycle(failing)
    for d in decs:
        flag = ">>> " if d.action in ("alert","reroute") else "    "
        print(f"  {flag}{d.node_id}: {d.action.upper():<8}  "
              f"sev={d.severity:.0f}  {d.reason}")
    print(f"\n  Quarantined: {pm.quarantined_nodes}")


# ============================================================
#  STEP 6 — Routing demo
# ============================================================
if not ONLY_STEP or ONLY_STEP == 6:
    banner("Step 6 — Dynamic Routing demo")
    router = DynamicRoutingAgent()
    topo = [
        {"node_id":"N01","battery_v":3.9, "rssi_dbm":-65, "pkt_loss_pct":1.2,
         "temp_c":38,"uptime_hrs":48, "hop_count":2,"queue_len_pkts":3, "tx_rate_kbps":88},
        {"node_id":"N02","battery_v":3.8, "rssi_dbm":-70, "pkt_loss_pct":2.1,
         "temp_c":41,"uptime_hrs":72, "hop_count":3,"queue_len_pkts":5, "tx_rate_kbps":82},
        {"node_id":"N03","battery_v":3.1, "rssi_dbm":-102,"pkt_loss_pct":38.0,
         "temp_c":69,"uptime_hrs":0.1,"hop_count":8,"queue_len_pkts":25,"tx_rate_kbps":7},
        {"node_id":"N04","battery_v":3.85,"rssi_dbm":-67, "pkt_loss_pct":1.5,
         "temp_c":39,"uptime_hrs":96, "hop_count":2,"queue_len_pkts":4, "tx_rate_kbps":87},
        {"node_id":"N05","battery_v":4.0, "rssi_dbm":-63, "pkt_loss_pct":0.9,
         "temp_c":36,"uptime_hrs":120,"hop_count":1,"queue_len_pkts":2, "tx_rate_kbps":95},
    ]
    router.update_topology(topo, {"N03"})
    router.compute_all_routes("HQ")
    router.print_route_table({"N03"})


# ============================================================
#  STEP 7 — Full orchestrator
# ============================================================
if not ONLY_STEP or ONLY_STEP == 7:
    banner(f"Step 7 — Full Orchestrator  ({MAX_CYCLES} cycles)")

    df_run = pd.read_csv(CSV_PATH)
    df_run.columns = (df_run.columns.str.strip().str.lower()
                      .str.replace(" ", "_").str.replace("%", "pct"))
    rename = {k.lower(): v for k, v in COLUMN_RENAME_MAP.items()}
    df_run = df_run.rename(columns=rename)
    if "timestamp" in df_run.columns:
        df_run["timestamp"] = pd.to_datetime(df_run["timestamp"])

    batches = [grp.to_dict("records")
               for _, grp in df_run.groupby("timestamp")]
    print(f"  Batches: {len(batches):,} | Running: {MAX_CYCLES}")

    pm2     = PredictiveMaintenanceAgent(MODEL_PATH)
    router2 = DynamicRoutingAgent()
    stats   = []
    prev    = None

    for cycle, batch in enumerate(batches[:MAX_CYCLES], 1):
        decs, summ = pm2.run_cycle(batch)
        if prev:
            pm2.reflect(batch)
        router2.update_topology(batch, pm2.quarantined_nodes)
        router2.compute_all_routes("HQ")
        ts = batch[0].get("timestamp", cycle)
        stats.append({
            "cycle": cycle, "ts": ts,
            "alert":  len(summ["alert"]),
            "reroute":len(summ["reroute"]),
            "watch":  len(summ["watch"]),
            "ok":     len(summ["ok"]),
            "quarantined": len(pm2.quarantined_nodes),
        })
        if summ["alert"] or summ["reroute"]:
            print(f"  Cycle {cycle:04d} | ALERT={summ['alert']} "
                  f"REROUTE={summ['reroute']} "
                  f"Q={list(pm2.quarantined_nodes)}")
        prev = batch

    stats_df = pd.DataFrame(stats)
    stats_df.to_csv("orchestrator_stats.csv", index=False)
    print(f"\n  Alerts  : {stats_df['alert'].sum()}")
    print(f"  Reroutes: {stats_df['reroute'].sum()}")
    print(f"  Stats   → orchestrator_stats.csv")


# ============================================================
#  STEP 8 — Visualise
# ============================================================
if not ONLY_STEP or ONLY_STEP == 8:
    banner("Step 8 — Visualize Results")

    if not os.path.exists("orchestrator_stats.csv"):
        print("  Run Step 7 first (orchestrator_stats.csv not found).")
    else:
        sdf = pd.read_csv("orchestrator_stats.csv")

        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        fig.suptitle(f"Disaster Mesh — Agentic AI ({len(sdf):,} cycles)",
                     fontsize=13, fontweight="500")

        axes[0].fill_between(sdf.cycle, sdf.alert,
                             alpha=0.85, color="#E24B4A", label="Alert")
        axes[0].fill_between(sdf.cycle, sdf.reroute,
                             alpha=0.65, color="#EF9F27", label="Reroute")
        axes[0].fill_between(sdf.cycle, sdf.watch,
                             alpha=0.45, color="#378ADD", label="Watch")
        axes[0].set_ylabel("Nodes")
        axes[0].set_title("PM Agent actions per cycle")
        axes[0].legend(loc="upper right")
        axes[0].set_ylim(0, 10)

        axes[1].fill_between(sdf.cycle, sdf.quarantined,
                             color="#A32D2D", alpha=0.7)
        axes[1].set_ylabel("Count")
        axes[1].set_title("Quarantined nodes")
        axes[1].set_ylim(0, 10)

        axes[2].fill_between(sdf.cycle, sdf.ok,
                             color="#1D9E75", alpha=0.7)
        axes[2].set_ylabel("Count")
        axes[2].set_xlabel("Cycle  (1 cycle = 5 min)")
        axes[2].set_title("Healthy nodes")
        axes[2].set_ylim(0, 10)

        plt.tight_layout()
        plt.savefig("agent_results.png", dpi=150, bbox_inches="tight")
        print("  Saved: agent_results.png")
        plt.show()

print("\n  All steps complete.")
