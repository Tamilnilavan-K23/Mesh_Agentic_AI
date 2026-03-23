# =============================================================
#  simulator.py — Dynamic real-time telemetry simulator
#  Generates per-node packets with 6 distinct scenarios.
#  Each scenario maps to a different mesh failure pattern.
# =============================================================

import numpy as np
import random
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict

# ── Scenario definitions ─────────────────────────────────────

SCENARIOS = {
    "normal": {
        "label": "Normal Operation",
        "description": "All 10 nodes healthy — baseline mesh performance.",
        "color": "#1D9E75",
        "icon": "✅",
        "failing_nodes": [],
        "degrading_nodes": [],
        "cascade": False,
    },
    "single_failure": {
        "label": "Single Node Failure",
        "description": "N07 battery draining — mesh reroutes around it.",
        "color": "#EF9F27",
        "icon": "⚠️",
        "failing_nodes": ["N07"],
        "degrading_nodes": ["N07"],
        "cascade": False,
    },
    "multi_failure": {
        "label": "Multi-Node Failure",
        "description": "N03 and N08 failing simultaneously — mesh under stress.",
        "color": "#E24B4A",
        "icon": "🔴",
        "failing_nodes": ["N03", "N08"],
        "degrading_nodes": ["N03", "N06", "N08"],
        "cascade": False,
    },
    "cascade_failure": {
        "label": "Cascade Failure",
        "description": "N02 fails → N05 overloads → N09 degrades. Satellite escalation triggered.",
        "color": "#A32D2D",
        "icon": "🔥",
        "failing_nodes": ["N02", "N05"],
        "degrading_nodes": ["N02", "N05", "N09"],
        "cascade": True,
    },
    "flood_scenario": {
        "label": "Flood Disaster",
        "description": "7 nodes submerged — only N01, N04, N10 survive. SOS routing critical.",
        "color": "#185FA5",
        "icon": "🌊",
        "failing_nodes": ["N02", "N03", "N05", "N06", "N07", "N08", "N09"],
        "degrading_nodes": ["N02", "N03", "N05", "N06", "N07", "N08", "N09"],
        "cascade": True,
    },
    "recovery": {
        "label": "Post-Failure Recovery",
        "description": "Nodes recovering after reroute — agent validates health restoration.",
        "color": "#534AB7",
        "icon": "🔄",
        "failing_nodes": [],
        "degrading_nodes": ["N03"],
        "cascade": False,
    },
}

NODES = [f"N{i:02d}" for i in range(1, 11)]


@dataclass
class NodeProfile:
    """Tracks the evolving state of one mesh node."""
    node_id: str
    step: int = 0
    uptime_hrs: float = field(default_factory=lambda: random.uniform(10, 200))
    base_battery: float = field(default_factory=lambda: random.uniform(3.7, 4.1))
    base_rssi: float = field(default_factory=lambda: random.uniform(-60, -72))

    def tick(self):
        self.step += 1
        self.uptime_hrs += 5 / 60  # 5-min interval


class ScenarioSimulator:
    """
    Generates realistic telemetry batches for a given scenario.
    Call next_batch() to get one timestep's readings for all 10 nodes.
    """

    def __init__(self, scenario_key: str = "normal", seed: int = None):
        self.scenario_key = scenario_key
        self.scenario     = SCENARIOS[scenario_key]
        self.profiles     = {nid: NodeProfile(nid) for nid in NODES}
        self.global_step  = 0
        self.rng          = np.random.default_rng(seed)
        self._rng_py      = random.Random(seed)

    def switch_scenario(self, new_key: str):
        """Hot-swap scenario without resetting node histories."""
        self.scenario_key = new_key
        self.scenario     = SCENARIOS[new_key]
        # Reset uptime for newly failing nodes so crash is visible
        for nid in self.scenario["failing_nodes"]:
            self.profiles[nid].step = 0

    def next_batch(self) -> List[dict]:
        """Return one batch of telemetry (one packet per node)."""
        self.global_step += 1
        batch = []
        for nid in NODES:
            prof = self.profiles[nid]
            prof.tick()
            pkt = self._generate_packet(nid, prof)
            batch.append(pkt)
        return batch

    # ── Packet generation ─────────────────────────────────────

    def _generate_packet(self, nid: str, prof: NodeProfile) -> dict:
        failing   = nid in self.scenario["failing_nodes"]
        degrading = nid in self.scenario["degrading_nodes"]
        cascade   = self.scenario["cascade"]

        # Degradation factor: ramps 0→1 over 20 steps for failing nodes
        deg = 0.0
        if failing:
            deg = min(1.0, prof.step / 20)
        elif degrading:
            deg = min(0.6, prof.step / 40)

        # Cascade: downstream nodes degrade slower
        if cascade and not failing and degrading:
            deg *= 0.5

        # Recovery scenario: nodes healing
        if self.scenario_key == "recovery" and degrading:
            deg = max(0.0, 0.5 - prof.step / 30)

        def nz(base, std, lo, hi):
            return float(np.clip(self.rng.normal(base, std), lo, hi))

        # Healthy baselines + degradation overlays
        battery_v      = nz(prof.base_battery - deg * 0.65, 0.04 + deg * 0.08, 2.9, 4.2)
        rssi_dbm       = nz(prof.base_rssi    - deg * 40,   3    + deg * 8,   -120, -40)
        pkt_loss_pct   = nz(1.0               + deg * 35,   0.4  + deg * 7,     0, 100)
        temp_c         = nz(37                + deg * 30,   1.5  + deg * 4,    20,  85)
        hop_count      = int(np.clip(self.rng.poisson(2 + deg * 5), 1, 10))
        queue_len_pkts = int(np.clip(self.rng.poisson(2 + deg * 20), 0, 50))
        tx_rate_kbps   = nz(85               - deg * 78,   4    + deg * 12,    1, 250)
        uptime_hrs     = prof.uptime_hrs if not (failing and deg > 0.8) else nz(0.15, 0.1, 0.05, 0.5)

        # 2% random noise spikes on healthy nodes
        if deg < 0.1 and self._rng_py.random() < 0.02:
            pkt_loss_pct = min(100, pkt_loss_pct + self._rng_py.uniform(5, 12))
            queue_len_pkts = min(50, queue_len_pkts + self._rng_py.randint(3, 7))

        return {
            "timestamp":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "node_id":        nid,
            "battery_v":      round(battery_v,      3),
            "rssi_dbm":       round(rssi_dbm,        1),
            "pkt_loss_pct":   round(pkt_loss_pct,    2),
            "temp_c":         round(temp_c,           1),
            "uptime_hrs":     round(uptime_hrs,       2),
            "hop_count":      hop_count,
            "queue_len_pkts": queue_len_pkts,
            "tx_rate_kbps":   round(tx_rate_kbps,    1),
            "deg_factor":     round(deg, 3),   # internal — not fed to agent
        }
