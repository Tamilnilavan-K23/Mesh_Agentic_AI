# =============================================================
#  step8_visualize.py
#  Generates all result charts from orchestrator_stats.csv.
#  Run AFTER step7_orchestrator.py.
#
#  Usage:
#      python step8_visualize.py
# =============================================================

import os
import sys
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Use non-interactive backend if no display (e.g. SSH session)
if not os.environ.get("DISPLAY") and sys.platform != "win32":
    matplotlib.use("Agg")

STATS_FILE = "orchestrator_stats.csv"

print("=" * 55)
print("  Step 8 — Visualize Results")
print("=" * 55)

if not os.path.exists(STATS_FILE):
    print(f"  ERROR: {STATS_FILE} not found.")
    print("  Run step7_orchestrator.py first.")
    sys.exit(1)

stats_df = pd.read_csv(STATS_FILE)
print(f"  Loaded {len(stats_df):,} cycles from {STATS_FILE}")
print(f"  Total alerts   : {stats_df['alert'].sum()}")
print(f"  Total reroutes : {stats_df['reroute'].sum()}")
print()


# ── Chart 1: Action timeline ────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
fig.suptitle(
    f"Disaster Mesh — Agentic AI  ({len(stats_df):,} cycles)",
    fontsize=13, fontweight="500"
)

ax0 = axes[0]
ax0.fill_between(stats_df.cycle, stats_df.alert,
                 alpha=0.85, color="#E24B4A", label="Alert (critical)")
ax0.fill_between(stats_df.cycle, stats_df.reroute,
                 alpha=0.65, color="#EF9F27", label="Reroute (warning)")
ax0.fill_between(stats_df.cycle, stats_df.watch,
                 alpha=0.45, color="#378ADD", label="Watch (trending)")
ax0.set_ylabel("Node count")
ax0.set_title("PM Agent — actions per cycle")
ax0.legend(loc="upper right", fontsize=9)
ax0.set_ylim(0, max(10, stats_df[["alert","reroute","watch"]].max().max() + 2))

ax1 = axes[1]
ax1.fill_between(stats_df.cycle, stats_df.quarantined,
                 color="#A32D2D", alpha=0.7)
ax1.set_ylabel("Node count")
ax1.set_title("Quarantined nodes (removed from routing)")
ax1.set_ylim(0, max(10, stats_df.quarantined.max() + 2))

ax2 = axes[2]
ax2.fill_between(stats_df.cycle, stats_df.ok, color="#1D9E75", alpha=0.7)
ax2.set_ylabel("Node count")
ax2.set_xlabel("Cycle number  (1 cycle = 5 min)")
ax2.set_title("Healthy nodes — no action required")
ax2.set_ylim(0, max(10, stats_df.ok.max() + 2))

plt.tight_layout()
plt.savefig("agent_results.png", dpi=150, bbox_inches="tight")
print("  Saved: agent_results.png")
plt.show()


# ── Chart 2: Per-node breakdown (needs decision log) ────────
dec_file = "decision_log.csv"

if os.path.exists(dec_file):
    dec_df = pd.read_csv(dec_file)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Per-node action breakdown", fontsize=12, fontweight="500")

    ac   = dec_df.groupby(["node_id", "action"]).size().unstack(fill_value=0)
    cols = [c for c in ["ok","watch","reroute","alert"] if c in ac.columns]
    clr  = {"ok":"#1D9E75","watch":"#378ADD","reroute":"#EF9F27","alert":"#E24B4A"}
    ac[cols].plot.bar(
        ax=axes[0],
        color=[clr[c] for c in cols],
        width=0.75,
        edgecolor="white",
    )
    axes[0].set_title("Actions per node")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=0)
    axes[0].legend(title="Action")

    non_ok = dec_df[dec_df.action != "ok"]
    if len(non_ok):
        non_ok["severity"].hist(
            ax=axes[1], bins=20,
            color="#7F77DD", edgecolor="white"
        )
    else:
        axes[1].text(0.5, 0.5,
                     "No failure events in this run.\n"
                     "Try --cycles 2000 for full coverage.",
                     ha="center", va="center",
                     transform=axes[1].transAxes)
    axes[1].set_title("Severity score distribution (non-ok)")
    axes[1].set_xlabel("Severity  (0 – 100)")
    axes[1].set_ylabel("Frequency")

    plt.tight_layout()
    plt.savefig("node_breakdown.png", dpi=150, bbox_inches="tight")
    print("  Saved: node_breakdown.png")
    plt.show()
else:
    print(f"  Skipping node breakdown chart "
          f"({dec_file} not found — decision log not exported by orchestrator).")


# ── Chart 3: Cumulative events ──────────────────────────────
fig, ax = plt.subplots(figsize=(12, 4))
fig.suptitle("Cumulative alerts and reroutes over time",
             fontsize=12, fontweight="500")
ax.plot(stats_df.cycle, stats_df.alert.cumsum(),
        color="#E24B4A", linewidth=1.5, label="Cumulative alerts")
ax.plot(stats_df.cycle, stats_df.reroute.cumsum(),
        color="#EF9F27", linewidth=1.5, label="Cumulative reroutes")
ax.set_xlabel("Cycle  (1 cycle = 5 min)")
ax.set_ylabel("Cumulative count")
ax.legend()
plt.tight_layout()
plt.savefig("cumulative_events.png", dpi=150, bbox_inches="tight")
print("  Saved: cumulative_events.png")
plt.show()

print("\n  All charts generated.")
print("  Files: agent_results.png  |  cumulative_events.png")
