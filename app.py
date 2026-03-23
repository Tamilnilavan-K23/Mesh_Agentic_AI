# =============================================================
#  app.py — Disaster Mesh · Agentic AI · Investor Dashboard
#
#  Usage:
#      streamlit run app.py
#
#  Requires: streamlit plotly pandas numpy scikit-learn joblib
# =============================================================

import time
import sys
import os
import threading
import queue
import copy
from collections import deque
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── local modules ─────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from simulator import ScenarioSimulator, SCENARIOS, NODES
from config import MODEL_PATH
from predictive_maintenance_agent import PredictiveMaintenanceAgent
from dynamic_routing_agent import DynamicRoutingAgent

# ═══════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="DisasterMesh · Agentic AI",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════
#  CUSTOM CSS — Dark industrial + signal-monitor aesthetic
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg:       #0b0e14;
    --surface:  #121620;
    --surface2: #1a2030;
    --border:   #2a3550;
    --accent:   #00d4aa;
    --accent2:  #4a9eff;
    --warn:     #f59e0b;
    --danger:   #ef4444;
    --ok:       #10b981;
    --purple:   #8b5cf6;
    --text:     #e2e8f0;
    --muted:    #64748b;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] * { color: var(--text) !important; }

h1, h2, h3 {
    font-family: 'Space Mono', monospace !important;
    letter-spacing: -0.02em;
}

.stButton > button {
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    color: #000 !important;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    border: none;
    border-radius: 6px;
    padding: 0.5rem 1.5rem;
    font-size: 13px;
    letter-spacing: 0.05em;
    cursor: pointer;
    width: 100%;
}
.stButton > button:hover { opacity: 0.85; }

.stSelectbox > div, .stSlider > div { color: var(--text) !important; }

/* Metric boxes */
.metric-box {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 10px;
    position: relative;
    overflow: hidden;
}
.metric-box::before {
    content:'';
    position:absolute;
    top:0; left:0; right:0;
    height:2px;
}
.metric-box.ok::before     { background: var(--ok); }
.metric-box.warn::before   { background: var(--warn); }
.metric-box.danger::before { background: var(--danger); }
.metric-box.info::before   { background: var(--accent2); }

.metric-label { font-size:11px; font-weight:600; letter-spacing:.1em;
                text-transform:uppercase; color:var(--muted); margin-bottom:4px; }
.metric-value { font-family:'Space Mono',monospace; font-size:26px;
                font-weight:700; color:var(--text); line-height:1; }
.metric-sub   { font-size:12px; color:var(--muted); margin-top:4px; }

/* Node status pill */
.node-pill {
    display:inline-flex; align-items:center; gap:6px;
    padding:4px 10px; border-radius:20px;
    font-size:12px; font-family:'Space Mono',monospace;
    font-weight:700; margin:3px;
}
.node-pill.ok      { background:#064e3b; color:#6ee7b7; border:1px solid #065f46; }
.node-pill.watch   { background:#1e3a5f; color:#93c5fd; border:1px solid #1d4ed8; }
.node-pill.reroute { background:#451a03; color:#fcd34d; border:1px solid #92400e; }
.node-pill.alert   { background:#450a0a; color:#fca5a5; border:1px solid #991b1b; }
.node-pill.dot     { width:7px; height:7px; border-radius:50%; display:inline-block; }
.dot-ok      { background:#10b981; }
.dot-watch   { background:#3b82f6; }
.dot-reroute { background:#f59e0b; }
.dot-alert   { background:#ef4444; box-shadow:0 0 6px #ef4444; }

/* Log entry */
.log-entry {
    font-family:'Space Mono',monospace; font-size:11px;
    padding:5px 10px; border-left:3px solid;
    margin-bottom:3px; border-radius:0 4px 4px 0;
    background: var(--surface2);
}
.log-entry.ok     { border-color:var(--ok);     color:#6ee7b7; }
.log-entry.watch  { border-color:var(--accent2); color:#93c5fd; }
.log-entry.reroute{ border-color:var(--warn);    color:#fcd34d; }
.log-entry.alert  { border-color:var(--danger);  color:#fca5a5; }
.log-entry.info   { border-color:var(--muted);   color:var(--muted); }

/* Scenario card */
.scenario-header {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 18px 24px;
    margin-bottom: 18px;
    display: flex; align-items: center; gap: 14px;
}
.scenario-icon  { font-size: 32px; }
.scenario-title { font-family:'Space Mono',monospace; font-size:16px;
                  font-weight:700; color:var(--text); margin:0; }
.scenario-desc  { font-size:13px; color:var(--muted); margin-top:3px; }

/* Route table */
.route-row {
    display:grid; grid-template-columns:60px 1fr 50px 60px 80px 70px;
    gap:8px; align-items:center;
    padding:8px 12px; border-radius:6px; margin-bottom:4px;
    font-size:12px; background:var(--surface2);
    border: 1px solid var(--border);
    font-family:'Space Mono',monospace;
}
.route-row.quarantined { opacity:.45; border-color:#ef4444; }

/* Satellite alert banner */
.sat-banner {
    background: linear-gradient(90deg, #450a0a, #7f1d1d);
    border: 1px solid #ef4444;
    border-radius: 8px; padding: 12px 20px;
    font-family:'Space Mono',monospace; font-size:13px;
    color:#fca5a5; font-weight:700;
    animation: pulse-border 1.5s ease-in-out infinite;
    margin-bottom: 12px;
}
@keyframes pulse-border {
    0%,100% { box-shadow: 0 0 8px #ef444466; }
    50%      { box-shadow: 0 0 18px #ef4444cc; }
}

/* Section titles */
.section-title {
    font-family:'Space Mono',monospace;
    font-size:11px; font-weight:700;
    letter-spacing:.15em; text-transform:uppercase;
    color:var(--muted); margin:18px 0 10px;
    border-bottom:1px solid var(--border);
    padding-bottom:6px;
}

[data-testid="stPlotlyChart"] { border-radius:10px; overflow:hidden; }

div[data-testid="column"] { gap: 0.5rem; }

.stAlert { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
#  SESSION STATE INIT
# ═══════════════════════════════════════════════════════════════
def _init_state():
    defaults = {
        "running":         False,
        "cycle":           0,
        "scenario_key":    "normal",
        "speed":           1.0,
        "history":         {nid: deque(maxlen=60) for nid in NODES},
        "action_log":      deque(maxlen=100),
        "stats":           deque(maxlen=200),
        "satellite":       False,
        "total_alerts":    0,
        "total_reroutes":  0,
        "total_watches":   0,
        "last_batch":      [],
        "last_decisions":  [],
        "last_routes":     [],
        "pm_agent":        None,
        "router":          None,
        "simulator":       None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ═══════════════════════════════════════════════════════════════
#  AGENT + SIMULATOR INIT
# ═══════════════════════════════════════════════════════════════
@st.cache_resource
def load_agents():
    pm     = PredictiveMaintenanceAgent(MODEL_PATH)
    router = DynamicRoutingAgent()
    return pm, router

def get_simulator(key):
    if (st.session_state.simulator is None or
            st.session_state.simulator.scenario_key != key):
        st.session_state.simulator = ScenarioSimulator(key, seed=42)
    return st.session_state.simulator

# ═══════════════════════════════════════════════════════════════
#  COLOURS HELPERS
# ═══════════════════════════════════════════════════════════════
ACTION_COLOR = {
    "ok":      "#10b981",
    "watch":   "#3b82f6",
    "reroute": "#f59e0b",
    "alert":   "#ef4444",
}
FEAT_LABELS = {
    "battery_v":      "Battery (V)",
    "rssi_dbm":       "RSSI (dBm)",
    "pkt_loss_pct":   "Pkt Loss (%)",
    "temp_c":         "Temp (°C)",
    "queue_len_pkts": "Queue (pkts)",
    "tx_rate_kbps":   "Tx Rate (kbps)",
}

def action_pill(nid, action):
    dot_cls = f"dot-{action}"
    return (f'<span class="node-pill {action}">'
            f'<span class="node-pill dot {dot_cls}"></span>{nid}</span>')

# ═══════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="padding:10px 0 20px">
      <div style="font-family:'Space Mono',monospace;font-size:18px;
                  font-weight:700;color:#00d4aa;letter-spacing:-.02em">
        🛰️ DisasterMesh
      </div>
      <div style="font-size:11px;color:#64748b;letter-spacing:.1em;
                  text-transform:uppercase;margin-top:2px">
        Agentic AI · Investor Demo
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Scenario</div>', unsafe_allow_html=True)
    scenario_key = st.selectbox(
        "Select scenario",
        options=list(SCENARIOS.keys()),
        format_func=lambda k: f"{SCENARIOS[k]['icon']}  {SCENARIOS[k]['label']}",
        key="scenario_select",
        label_visibility="collapsed",
    )

    sc = SCENARIOS[scenario_key]
    st.markdown(f"""
    <div style="background:#1a2030;border:1px solid #2a3550;border-radius:8px;
                padding:12px 14px;margin:8px 0;font-size:12px;color:#94a3b8">
        {sc['description']}
    </div>""", unsafe_allow_html=True)

    if sc["failing_nodes"]:
        st.markdown(
            f'<div style="font-size:11px;color:#ef4444;margin-bottom:8px">'
            f'⚡ Failing: {", ".join(sc["failing_nodes"])}</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-title">Simulation Speed</div>', unsafe_allow_html=True)
    speed = st.slider("Interval (s)", 0.3, 3.0, 1.0, 0.1,
                      label_visibility="collapsed")

    st.markdown('<div class="section-title">Controls</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        start_btn = st.button("▶  START", key="start")
    with col_b:
        stop_btn  = st.button("⏸  STOP",  key="stop")

    reset_btn = st.button("↺  RESET ALL", key="reset")

    st.markdown('<div class="section-title">Session Stats</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-family:'Space Mono',monospace;font-size:12px;
                color:#94a3b8;line-height:2">
      Cycle &nbsp;&nbsp;&nbsp;: <span style="color:#e2e8f0">{st.session_state.cycle}</span><br>
      Alerts &nbsp;&nbsp;: <span style="color:#ef4444">{st.session_state.total_alerts}</span><br>
      Reroutes : <span style="color:#f59e0b">{st.session_state.total_reroutes}</span><br>
      Watches &nbsp;: <span style="color:#3b82f6">{st.session_state.total_watches}</span>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">About</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:11px;color:#64748b;line-height:1.7">
      Agentic AI predicts mesh node<br>
      failures before they happen and<br>
      reroutes SOS traffic in real time.<br><br>
      <b style="color:#00d4aa">6 disaster scenarios</b> · 10 nodes<br>
      Observe→Reason→Plan→Act loop<br>
      Random Forest (93% test acc.)
    </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
#  HANDLE CONTROLS
# ═══════════════════════════════════════════════════════════════
if start_btn:
    st.session_state.running = True
    if st.session_state.scenario_key != scenario_key:
        st.session_state.scenario_key = scenario_key
        sim = get_simulator(scenario_key)
        sim.switch_scenario(scenario_key)

if stop_btn:
    st.session_state.running = False

if reset_btn:
    for k in ["cycle","total_alerts","total_reroutes","total_watches",
              "satellite","last_batch","last_decisions","last_routes"]:
        if k in ["last_batch","last_decisions","last_routes"]:
            st.session_state[k] = []
        elif k == "satellite":
            st.session_state[k] = False
        else:
            st.session_state[k] = 0
    st.session_state.history  = {nid: deque(maxlen=60) for nid in NODES}
    st.session_state.action_log = deque(maxlen=100)
    st.session_state.stats      = deque(maxlen=200)
    st.session_state.running    = False
    st.session_state.simulator  = None
    st.session_state.pm_agent   = None
    st.session_state.router     = None
    st.rerun()

# ── Scenario hot-swap ─────────────────────────────────────────
if scenario_key != st.session_state.scenario_key:
    st.session_state.scenario_key = scenario_key
    sim = st.session_state.simulator
    if sim:
        sim.switch_scenario(scenario_key)

# ═══════════════════════════════════════════════════════════════
#  MAIN HEADER
# ═══════════════════════════════════════════════════════════════
sc_now = SCENARIOS[st.session_state.scenario_key]

st.markdown(f"""
<div style="display:flex;align-items:center;gap:16px;
            border-bottom:1px solid #2a3550;padding-bottom:16px;margin-bottom:20px">
  <div style="font-size:40px">{sc_now['icon']}</div>
  <div>
    <h1 style="margin:0;font-size:22px;color:#e2e8f0">
      Disaster Mesh  <span style="color:#00d4aa">· Agentic AI</span>
    </h1>
    <div style="font-size:13px;color:#64748b;margin-top:2px">
      Real-time predictive maintenance + dynamic routing · 10-node LoRa mesh
    </div>
  </div>
  <div style="margin-left:auto;text-align:right">
    <div style="font-family:'Space Mono',monospace;font-size:11px;color:#64748b">
      SCENARIO
    </div>
    <div style="font-family:'Space Mono',monospace;font-size:14px;
                color:{sc_now['color']};font-weight:700">
      {sc_now['label'].upper()}
    </div>
    <div style="font-family:'Space Mono',monospace;font-size:11px;
                color:{'#ef4444' if st.session_state.running else '#64748b'}">
      {'● LIVE' if st.session_state.running else '○ PAUSED'}
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
#  RUN ONE SIMULATION CYCLE
# ═══════════════════════════════════════════════════════════════

def _hex_rgba(hex_color, alpha=0.15):
    """Convert #rrggbb hex to rgba() string — Plotly compatible."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return f"rgba({r},{g},{b},{alpha})"

def run_one_cycle():
    # Init agents once
    if st.session_state.pm_agent is None:
        pm, router = load_agents()
        st.session_state.pm_agent = pm
        st.session_state.router   = router
    pm     = st.session_state.pm_agent
    router = st.session_state.router
    sim    = get_simulator(st.session_state.scenario_key)

    # Generate batch
    batch = sim.next_batch()

    # Strip internal field before feeding to agent
    clean_batch = [{k:v for k,v in pkt.items() if k != "deg_factor"}
                   for pkt in batch]

    # Run PM agent
    decisions, summary = pm.run_cycle(clean_batch)

    # Satellite check
    fail_frac = len(summary["alert"]) / max(len(NODES), 1)
    sat = fail_frac >= 0.40
    st.session_state.satellite = sat

    # Run routing agent
    router.update_topology(clean_batch, pm.quarantined_nodes)
    router.compute_all_routes("HQ")
    route_table = router.get_route_table()

    # Update histories
    for pkt in batch:
        nid = pkt["node_id"]
        st.session_state.history[nid].append(pkt)

    # Update stats
    st.session_state.cycle += 1
    st.session_state.total_alerts    += len(summary["alert"])
    st.session_state.total_reroutes  += len(summary["reroute"])
    st.session_state.total_watches   += len(summary["watch"])
    st.session_state.stats.append({
        "cycle":      st.session_state.cycle,
        "alert":      len(summary["alert"]),
        "reroute":    len(summary["reroute"]),
        "watch":      len(summary["watch"]),
        "ok":         len(summary["ok"]),
        "quarantined":len(pm.quarantined_nodes),
    })

    # Build action log entries
    ts = datetime.now().strftime("%H:%M:%S")
    for d in decisions:
        if d.action != "ok":
            reason_short = d.reason[:55] + "…" if len(d.reason) > 55 else d.reason
            st.session_state.action_log.appendleft({
                "ts":     ts,
                "node":   d.node_id,
                "action": d.action,
                "reason": reason_short,
                "sev":    d.severity,
            })
    if not any(d.action != "ok" for d in decisions):
        st.session_state.action_log.appendleft({
            "ts": ts, "node": "ALL", "action": "ok",
            "reason": f"All {len(NODES)} nodes healthy", "sev": 0,
        })

    if sat:
        st.session_state.action_log.appendleft({
            "ts": ts, "node": "SYS", "action": "alert",
            "reason": f"SATELLITE ESCALATION — {fail_frac:.0%} nodes failing",
            "sev": 100,
        })

    st.session_state.last_batch     = batch
    st.session_state.last_decisions = decisions
    st.session_state.last_routes    = route_table

    return summary, route_table

if st.session_state.running:
    summary, route_table = run_one_cycle()
else:
    summary      = {"alert":[],"reroute":[],"watch":[],"ok":list(NODES)}
    route_table  = st.session_state.last_routes

# ═══════════════════════════════════════════════════════════════
#  SATELLITE BANNER
# ═══════════════════════════════════════════════════════════════
if st.session_state.satellite:
    st.markdown(
        '<div class="sat-banner">🛰️  SATELLITE FALLBACK ACTIVATED — '
        'Critical mass of nodes failing. SOS traffic rerouted via Iridium gateway.</div>',
        unsafe_allow_html=True,
    )

# ═══════════════════════════════════════════════════════════════
#  ROW 1 — KPI METRICS
# ═══════════════════════════════════════════════════════════════
pm_agent_now = st.session_state.pm_agent
quarantined  = pm_agent_now.quarantined_nodes if pm_agent_now else set()
n_active     = len(NODES) - len(quarantined)
n_alert      = len(summary["alert"])
n_reroute    = len(summary["reroute"])
n_watch      = len(summary["watch"])

c1, c2, c3, c4, c5, c6 = st.columns(6)

def metric_html(label, value, sub, cls):
    return f"""
    <div class="metric-box {cls}">
      <div class="metric-label">{label}</div>
      <div class="metric-value">{value}</div>
      <div class="metric-sub">{sub}</div>
    </div>"""

with c1:
    st.markdown(metric_html(
        "Active Nodes", n_active, f"{len(quarantined)} quarantined", "ok"
    ), unsafe_allow_html=True)
with c2:
    cls = "danger" if n_alert else "ok"
    st.markdown(metric_html(
        "Critical Alerts", n_alert, f"Total: {st.session_state.total_alerts}", cls
    ), unsafe_allow_html=True)
with c3:
    cls = "warn" if n_reroute else "ok"
    st.markdown(metric_html(
        "Reroutes", n_reroute, f"Total: {st.session_state.total_reroutes}", cls
    ), unsafe_allow_html=True)
with c4:
    cls = "info" if n_watch else "ok"
    st.markdown(metric_html(
        "Watching", n_watch, "Trending bad", cls
    ), unsafe_allow_html=True)
with c5:
    st.markdown(metric_html(
        "Cycle", st.session_state.cycle,
        f"≈ {st.session_state.cycle * 5} min elapsed", "info"
    ), unsafe_allow_html=True)
with c6:
    status_txt = "LIVE" if st.session_state.running else "PAUSED"
    cls = "ok" if st.session_state.running else "warn"
    st.markdown(metric_html(
        "Status", status_txt, sc_now["label"], cls
    ), unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
#  ROW 2 — Node grid + Trend chart
# ═══════════════════════════════════════════════════════════════
col_nodes, col_trend = st.columns([1, 2])

with col_nodes:
    st.markdown('<div class="section-title">Node Status Grid</div>',
                unsafe_allow_html=True)
    d_map = {d.node_id: d for d in st.session_state.last_decisions}
    pills = ""
    for nid in NODES:
        dec   = d_map.get(nid)
        action= dec.action if dec else "ok"
        pills += action_pill(nid, action)
    st.markdown(
        f'<div style="line-height:2.2">{pills}</div>',
        unsafe_allow_html=True,
    )

    # Mini feature readout for last batch
    if st.session_state.last_batch:
        st.markdown('<div class="section-title">Latest Reading — '
                    'Top Stressed Node</div>', unsafe_allow_html=True)
        worst = max(
            st.session_state.last_decisions,
            key=lambda d: d.severity,
            default=None,
        ) if st.session_state.last_decisions else None

        if worst:
            pkt = next((p for p in st.session_state.last_batch
                        if p["node_id"] == worst.node_id), None)
            if pkt:
                rows = ""
                for feat, lbl in FEAT_LABELS.items():
                    val = pkt.get(feat, 0)
                    color = "#ef4444" if feat in (worst.top_features or []) else "#94a3b8"
                    rows += (f'<div style="display:flex;justify-content:space-between;'
                             f'padding:3px 0;font-size:12px;border-bottom:1px solid #2a3550">'
                             f'<span style="color:#64748b">{lbl}</span>'
                             f'<span style="color:{color};font-family:Space Mono,monospace'
                             f';font-weight:700">{val}</span></div>')
                st.markdown(
                    f'<div style="background:#1a2030;border-radius:8px;'
                    f'padding:10px 14px;border:1px solid #2a3550">'
                    f'<div style="font-family:Space Mono,monospace;font-size:12px;'
                    f'color:#00d4aa;margin-bottom:6px">{worst.node_id} — '
                    f'sev {worst.severity:.0f}/100</div>{rows}</div>',
                    unsafe_allow_html=True,
                )

with col_trend:
    st.markdown('<div class="section-title">Action Timeline</div>',
                unsafe_allow_html=True)
    if st.session_state.stats:
        sdf   = pd.DataFrame(list(st.session_state.stats))
        fig   = go.Figure()
        for key, color, name in [
            ("alert",      "#ef4444", "Alert (critical)"),
            ("reroute",    "#f59e0b", "Reroute (warning)"),
            ("watch",      "#3b82f6", "Watch (trending)"),
            ("quarantined","#8b5cf6", "Quarantined"),
        ]:
            fig.add_trace(go.Scatter(
                x=sdf.cycle, y=sdf[key],
                fill="tozeroy", mode="lines",
                line=dict(color=color, width=1.5),
                fillcolor=_hex_rgba(color, 0.15),
                name=name,
            ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#121620",
            font=dict(family="Space Mono", size=10, color="#94a3b8"),
            margin=dict(l=0, r=0, t=4, b=0),
            height=230,
            legend=dict(orientation="h", y=-0.15, x=0,
                        bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
            xaxis=dict(gridcolor="#1a2030", title="Cycle", tickfont=dict(size=9)),
            yaxis=dict(gridcolor="#1a2030", title="Nodes", tickfont=dict(size=9)),
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
    else:
        st.markdown(
            '<div style="height:220px;display:flex;align-items:center;'
            'justify-content:center;color:#2a3550;font-family:Space Mono,monospace;'
            'font-size:13px">Start simulation to see live timeline</div>',
            unsafe_allow_html=True,
        )

# ═══════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════
#  ROW 3 — Mesh topology map + Routing table
# ═══════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">Live Mesh Network — Dynamic Routing Map</div>',
            unsafe_allow_html=True)

col_topo, col_route = st.columns([3, 2])

with col_topo:
    NODE_POS = {
        "N01": (0.15, 0.85), "N02": (0.50, 0.92), "N03": (0.85, 0.85),
        "N04": (0.05, 0.55), "N05": (0.35, 0.62), "N06": (0.65, 0.62),
        "N07": (0.95, 0.55), "N08": (0.20, 0.28), "N09": (0.60, 0.28),
        "N10": (0.40, 0.10),
    }
    HQ_POS = (0.50, -0.05)
    fig_topo = go.Figure()

    # Faint mesh background edges
    for nid_a, (xa, ya) in NODE_POS.items():
        for nid_b, (xb, yb) in NODE_POS.items():
            if nid_a >= nid_b:
                continue
            if ((xa-xb)**2 + (ya-yb)**2)**0.5 < 0.45:
                fig_topo.add_trace(go.Scatter(
                    x=[xa, xb], y=[ya, yb], mode="lines",
                    line=dict(color="rgba(42,53,80,0.8)", width=1),
                    hoverinfo="skip", showlegend=False,
                ))

    # HQ node
    fig_topo.add_trace(go.Scatter(
        x=[HQ_POS[0]], y=[HQ_POS[1]],
        mode="markers+text",
        marker=dict(size=28, color="#1a2030",
                    line=dict(color="#00d4aa", width=2.5), symbol="diamond"),
        text=["HQ"], textposition="top center",
        textfont=dict(size=10, color="#00d4aa", family="Space Mono"),
        hoverinfo="skip", showlegend=False,
    ))

    # Active routing paths
    if route_table:
        for r in route_table:
            if not r.get("available", True):
                continue
            path_nodes = r["path"].split(" -> ") if isinstance(r["path"], str) else r["path"]
            path_coords = []
            for step in path_nodes:
                step = step.strip()
                if step == "HQ":
                    path_coords.append(HQ_POS)
                elif step in NODE_POS:
                    path_coords.append(NODE_POS[step])
            if len(path_coords) < 2:
                continue
            conf_color = {"high":"rgba(0,212,170,0.75)",
                          "medium":"rgba(239,159,39,0.65)",
                          "low":"rgba(239,68,68,0.55)"}.get(r["confidence"],"rgba(100,116,139,0.4)")
            lw = {"high":2.8,"medium":1.8,"low":1.2}.get(r["confidence"],1.5)
            fig_topo.add_trace(go.Scatter(
                x=[c[0] for c in path_coords],
                y=[c[1] for c in path_coords],
                mode="lines",
                line=dict(color=conf_color, width=lw),
                hovertemplate=(
                    f"<b>{r['node_id']} → HQ</b><br>"
                    f"Path: {r['path']}<br>"
                    f"Hops: {r['hops']} | Score: {r['score']:.3f}<br>"
                    f"Latency: {r['latency_ms']:.0f}ms | Conf: {r['confidence']}"
                    "<extra></extra>"
                ),
                showlegend=False,
            ))

    # Nodes
    d_map_r = {d.node_id: d for d in st.session_state.last_decisions}
    for nid, (nx, ny) in NODE_POS.items():
        dec    = d_map_r.get(nid)
        action = dec.action if dec else "ok"
        sev    = dec.severity if dec else 0.0
        is_q   = nid in quarantined
        nc = {"ok":"#10b981","watch":"#3b82f6","reroute":"#f59e0b","alert":"#ef4444"}.get(action,"#64748b")
        top_f = (", ".join(dec.top_features) if dec and dec.top_features else "All OK")
        hover = (f"<b>{nid}</b><br>Action: {action.upper()}<br>"
                 f"Severity: {sev:.0f}/100<br>Breached: {top_f}"
                 + ("<br>QUARANTINED" if is_q else ""))
        fig_topo.add_trace(go.Scatter(
            x=[nx], y=[ny], mode="markers+text",
            marker=dict(size=20 + int(sev/10),
                        color="#450a0a" if is_q else nc,
                        line=dict(color="#ef4444" if is_q else nc, width=2.5),
                        symbol="x" if is_q else "circle",
                        opacity=0.35 if is_q else 1.0),
            text=[nid], textposition="top center",
            textfont=dict(size=9, color=nc, family="Space Mono"),
            hovertemplate=hover + "<extra></extra>",
            showlegend=False,
        ))

    # Legend
    for lbl, lc, sym in [("OK","#10b981","circle"),("Watch","#3b82f6","circle"),
                           ("Reroute","#f59e0b","circle"),("Alert/Q","#ef4444","x")]:
        fig_topo.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=10, color=lc, symbol=sym),
            name=lbl, showlegend=True,
        ))

    # Path confidence legend
    for lbl, lc in [("Path: high","rgba(0,212,170,0.9)"),
                     ("Path: medium","rgba(239,159,39,0.9)"),
                     ("Path: low","rgba(239,68,68,0.9)")]:
        fig_topo.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines",
            line=dict(color=lc, width=2),
            name=lbl, showlegend=True,
        ))

    fig_topo.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0d1117",
        font=dict(family="Space Mono", size=9, color="#94a3b8"),
        margin=dict(l=0, r=0, t=10, b=0),
        height=380,
        xaxis=dict(visible=False, range=[-0.05, 1.05]),
        yaxis=dict(visible=False, range=[-0.18, 1.08]),
        legend=dict(orientation="h", y=-0.04, x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=9)),
        hovermode="closest",
    )
    st.plotly_chart(fig_topo, use_container_width=True,
                    config={"displayModeBar": False})

with col_route:
    st.markdown('<div class="section-title">Routing Table → HQ</div>',
                unsafe_allow_html=True)
    if route_table:
        st.markdown(
            '<div class="route-row" style="font-size:10px;color:#64748b;'
            'background:transparent;border:none;padding-bottom:0">'
            '<span>NODE</span><span>PATH</span><span>HOPS</span>'
            '<span>SCORE</span><span>LAT(ms)</span><span>CONF</span></div>',
            unsafe_allow_html=True,
        )
        for r in route_table:
            is_q   = not r.get("available", True)
            q_cls  = "quarantined" if is_q else ""
            c_color = {"high":"#10b981","medium":"#f59e0b","low":"#ef4444"}.get(
                r["confidence"], "#94a3b8")
            nid_color = "#ef4444" if is_q else "#00d4aa"
            st.markdown(
                f'<div class="route-row {q_cls}">'
                f'<span style="color:{nid_color};font-weight:700">{r["node_id"]}</span>'
                f'<span style="color:#64748b;font-size:10px">{r["path"]}</span>'
                f'<span>{r["hops"]}</span>'
                f'<span>{r["score"]:.3f}</span>'
                f'<span>{r["latency_ms"]:.0f}</span>'
                f'<span style="color:{c_color}">{r["confidence"]}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<div style="color:#2a3550;font-family:Space Mono,monospace;'
            'font-size:12px;padding:20px 0">No routing data yet</div>',
            unsafe_allow_html=True,
        )

# =============================================================
# ═══════════════════════════════════════════════════════════════
#  ROW 4 — Agent decision log (full width)
# ═══════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">Agent Decision Log</div>',
            unsafe_allow_html=True)
log = list(st.session_state.action_log)[:20]
if log:
    for entry in log:
        action  = entry["action"]
        icon_map= {"ok":"✓","watch":"◎","reroute":"⚡","alert":"✖"}
        icon    = icon_map.get(action, "·")
        st.markdown(
            f'<div class="log-entry {action}">'
            f'<span style="opacity:.5">[{entry["ts"]}]</span>  '
            f'{icon} <b>{entry["node"]}</b>  {entry["reason"]}'
            f'</div>',
            unsafe_allow_html=True,
        )
else:
    st.markdown(
        '<div style="color:#2a3550;font-family:Space Mono,monospace;'
        'font-size:12px;padding:20px 0">Waiting for first cycle…</div>',
        unsafe_allow_html=True,
    )

# ═══════════════════════════════════════════════════════════════
#  ROW 5 — Scenario comparison radar + Report
# ═══════════════════════════════════════════════════════════════
col_radar, col_report = st.columns([1, 1])

with col_radar:
    st.markdown('<div class="section-title">Current Node Health Radar</div>',
                unsafe_allow_html=True)
    if st.session_state.last_batch:
        from config import FEATURE_RANGES
        radar_feats = ["battery_v","rssi_dbm","pkt_loss_pct",
                       "temp_c","hop_count","tx_rate_kbps"]
        radar_labels= ["Battery","RSSI","Pkt Loss","Temp","Hops","Tx Rate"]

        fig_rad = go.Figure()
        palette = ["#00d4aa","#4a9eff","#f59e0b","#ef4444",
                   "#8b5cf6","#10b981","#f97316","#06b6d4","#ec4899","#84cc16"]
        for idx, pkt in enumerate(st.session_state.last_batch):
            vals = []
            for feat in radar_feats:
                v = pkt.get(feat, 0)
                lo, hi = FEATURE_RANGES[feat]
                norm = (v - lo) / (hi - lo + 1e-9)
                if feat in ("rssi_dbm","battery_v","tx_rate_kbps"):
                    norm = 1 - norm
                vals.append(round(norm, 3))
            vals.append(vals[0])    # close polygon
            labels = radar_labels + [radar_labels[0]]
            nid = pkt["node_id"]
            is_q = nid in quarantined
            fig_rad.add_trace(go.Scatterpolar(
                r=vals, theta=labels,
                fill="toself",
                fillcolor=_hex_rgba(palette[idx % len(palette)], 0.09),
                line=dict(color=palette[idx % len(palette)],
                          width=1.2 if not is_q else 0.4,
                          dash="dot" if is_q else "solid"),
                name=nid,
                opacity=0.4 if is_q else 0.9,
            ))
        fig_rad.update_layout(
            polar=dict(
                bgcolor="#121620",
                radialaxis=dict(visible=True, range=[0,1],
                                gridcolor="#2a3550", tickfont=dict(size=8)),
                angularaxis=dict(gridcolor="#2a3550",
                                 tickfont=dict(size=9, color="#94a3b8")),
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Space Mono", size=9, color="#94a3b8"),
            legend=dict(orientation="h", y=-0.1, font=dict(size=8),
                        bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=0, r=0, t=4, b=0),
            height=300,
            showlegend=True,
        )
        st.plotly_chart(fig_rad, use_container_width=True,
                        config={"displayModeBar": False})

with col_report:
    st.markdown('<div class="section-title">Agentic AI — Session Report</div>',
                unsafe_allow_html=True)

    sc_info = SCENARIOS[st.session_state.scenario_key]
    cycles  = st.session_state.cycle
    uptime  = round(cycles * 5 / 60, 1)

    # Prevention rate calculation
    total_events = (st.session_state.total_alerts +
                    st.session_state.total_reroutes)
    _prev_pct = 0 if total_events == 0 else min(100, int(st.session_state.total_reroutes / max(total_events, 1) * 100))
    prevention_label = "N/A" if total_events == 0 else f"{_prev_pct}%"

    d_map_now = {d.node_id: d for d in st.session_state.last_decisions}
    node_summary_rows = ""
    for nid in NODES:
        dec = d_map_now.get(nid)
        action = dec.action if dec else "ok"
        sev    = f"{dec.severity:.0f}" if dec else "0"
        q_flag = " ⊗" if nid in quarantined else ""
        color  = ACTION_COLOR.get(action, "#94a3b8")
        node_summary_rows += (
            f'<div style="display:flex;justify-content:space-between;'
            f'padding:3px 0;font-size:11px;border-bottom:1px solid #1a2030;'
            f'font-family:Space Mono,monospace">'
            f'<span style="color:#94a3b8">{nid}{q_flag}</span>'
            f'<span style="color:{color};font-weight:700">{action.upper()}</span>'
            f'<span style="color:#64748b">sev {sev}</span></div>'
        )

    st.markdown(f"""
    <div style="background:#121620;border:1px solid #2a3550;
                border-radius:10px;padding:16px 18px;font-size:12px">
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:14px">
        <div style="background:#1a2030;border-radius:6px;padding:10px">
          <div style="color:#64748b;font-size:10px;text-transform:uppercase;letter-spacing:.1em">Scenario</div>
          <div style="color:{sc_info['color']};font-family:Space Mono,monospace;
                      font-weight:700;font-size:12px;margin-top:3px">{sc_info['label']}</div>
        </div>
        <div style="background:#1a2030;border-radius:6px;padding:10px">
          <div style="color:#64748b;font-size:10px;text-transform:uppercase;letter-spacing:.1em">Uptime monitored</div>
          <div style="color:#e2e8f0;font-family:Space Mono,monospace;
                      font-weight:700;font-size:12px;margin-top:3px">{uptime} hrs ({cycles} cycles)</div>
        </div>
        <div style="background:#1a2030;border-radius:6px;padding:10px">
          <div style="color:#64748b;font-size:10px;text-transform:uppercase;letter-spacing:.1em">Failures predicted</div>
          <div style="color:#ef4444;font-family:Space Mono,monospace;
                      font-weight:700;font-size:12px;margin-top:3px">{st.session_state.total_alerts} alerts</div>
        </div>
        <div style="background:#1a2030;border-radius:6px;padding:10px">
          <div style="color:#64748b;font-size:10px;text-transform:uppercase;letter-spacing:.1em">Traffic saved</div>
          <div style="color:#f59e0b;font-family:Space Mono,monospace;
                      font-weight:700;font-size:12px;margin-top:3px">{st.session_state.total_reroutes} reroutes</div>
        </div>
      </div>
      <div style="color:#64748b;font-size:10px;text-transform:uppercase;
                  letter-spacing:.1em;margin-bottom:6px">Current node decisions</div>
      {node_summary_rows}
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
#  AUTO-REFRESH
# ═══════════════════════════════════════════════════════════════
if st.session_state.running:
    time.sleep(speed)
    st.rerun()
else:
    st.markdown(
        '<div style="text-align:center;color:#2a3550;font-family:Space Mono,'
        'monospace;font-size:11px;padding:20px 0;letter-spacing:.1em">'
        'PRESS ▶ START TO BEGIN SIMULATION</div>',
        unsafe_allow_html=True,
    )