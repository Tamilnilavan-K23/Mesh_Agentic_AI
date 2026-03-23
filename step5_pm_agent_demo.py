# =============================================================
#  step5_pm_agent_demo.py
#  Run this first to verify the PM agent works on your model.
#  Tests two scenarios: all-healthy batch and one-failing batch.
#
#  Usage:
#      python step5_pm_agent_demo.py
# =============================================================

import sys
from config import MODEL_PATH
from predictive_maintenance_agent import PredictiveMaintenanceAgent

print("=" * 60)
print("  Step 5 — PM Agent Demo")
print("=" * 60)
print(f"  Model: {MODEL_PATH}\n")

pm = PredictiveMaintenanceAgent(MODEL_PATH)

# ── Scenario A: All healthy nodes ──────────────────────────
healthy_batch = [
    {
        "node_id": "N01", "battery_v": 3.9, "rssi_dbm": -65,
        "pkt_loss_pct": 1.2, "temp_c": 38, "uptime_hrs": 48,
        "hop_count": 2, "queue_len_pkts": 3, "tx_rate_kbps": 88,
    },
    {
        "node_id": "N02", "battery_v": 3.8, "rssi_dbm": -70,
        "pkt_loss_pct": 2.1, "temp_c": 41, "uptime_hrs": 72,
        "hop_count": 3, "queue_len_pkts": 5, "tx_rate_kbps": 82,
    },
    {
        "node_id": "N03", "battery_v": 4.0, "rssi_dbm": -62,
        "pkt_loss_pct": 0.8, "temp_c": 36, "uptime_hrs": 120,
        "hop_count": 1, "queue_len_pkts": 2, "tx_rate_kbps": 95,
    },
]

print("\n>>> Scenario A: All healthy nodes")
print("-" * 50)
decs, summ = pm.run_cycle(healthy_batch)
for d in decs:
    print(f"  {d.node_id}: {d.action.upper():<8} | {d.reason}")
print(f"\n  Summary  → {summ}")

# ── Scenario B: One critically failing node ─────────────────
failing_batch = [
    {
        "node_id": "N04", "battery_v": 3.1, "rssi_dbm": -102,
        "pkt_loss_pct": 38, "temp_c": 69, "uptime_hrs": 0.1,
        "hop_count": 8, "queue_len_pkts": 25, "tx_rate_kbps": 7,
    },
    {
        "node_id": "N05", "battery_v": 3.85, "rssi_dbm": -68,
        "pkt_loss_pct": 1.5, "temp_c": 40, "uptime_hrs": 96,
        "hop_count": 2, "queue_len_pkts": 4, "tx_rate_kbps": 86,
    },
]

print("\n>>> Scenario B: N04 critically failing")
print("-" * 50)
decs, summ = pm.run_cycle(failing_batch)
for d in decs:
    tag = ">>> FAIL" if d.action in ("alert", "reroute") else "    OK  "
    print(f"  {tag}  {d.node_id}: {d.action.upper():<8} "
          f"sev={d.severity:.0f}  | {d.reason}")

print(f"\n  Summary        → {summ}")
print(f"  Quarantined    → {pm.quarantined_nodes}")

# ── Scenario C: Reflect — did rerouting help? ──────────────
recovered_batch = [
    {
        "node_id": "N04", "battery_v": 3.7, "rssi_dbm": -75,
        "pkt_loss_pct": 4.0, "temp_c": 42, "uptime_hrs": 0.5,
        "hop_count": 3, "queue_len_pkts": 6, "tx_rate_kbps": 70,
    },
]

print("\n>>> Scenario C: Reflect — N04 after reroute")
print("-" * 50)
outcomes = pm.reflect(recovered_batch)
for nid, outcome in outcomes.items():
    print(f"  {nid}: {outcome.upper()}")
print(f"  Quarantined now → {pm.quarantined_nodes}")
print("\n  [Step 5 complete — run step6_routing_demo.py next]")
