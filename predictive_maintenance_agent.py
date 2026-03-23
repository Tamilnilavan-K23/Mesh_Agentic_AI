# =============================================================
#  predictive_maintenance_agent.py
#  6-stage agentic loop: Observe → Perceive → Reason →
#                        Plan → Act → Reflect
# =============================================================

import joblib
import numpy as np
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from config import (
    FEATURES, MODEL_PATH, MIN_CONFIDENCE,
    SEVERITY_CRITICAL, SEVERITY_WARNING, SEVERITY_WATCH,
    TREND_WINDOW, SATELLITE_THRESHOLD,
)
from preprocessor import TelemetryPreprocessor


# ----------------------------------------------------------
#  Data classes
# ----------------------------------------------------------

@dataclass
class NodeStatus:
    node_id:    str
    label:      int           # 0 = healthy, 1 = failure predicted
    confidence: float
    severity:   float         # 0–100 composite score
    breached:   list          # feature names that breached threshold
    action:     str           # "ok" | "watch" | "reroute" | "alert"
    timestamp:  datetime = field(default_factory=datetime.now)
    raw_packet: dict     = field(default_factory=dict)


@dataclass
class AgentDecision:
    node_id:      str
    action:       str
    reason:       str
    severity:     float
    top_features: list
    timestamp:    datetime      = field(default_factory=datetime.now)
    outcome:      Optional[str] = None   # filled in at Reflect stage


# ----------------------------------------------------------
#  Agent
# ----------------------------------------------------------

class PredictiveMaintenanceAgent:
    """
    Full Observe→Perceive→Reason→Plan→Act→Reflect loop.

    Usage:
        agent = PredictiveMaintenanceAgent("models/rf_model.pkl")
        decisions, summary = agent.run_cycle(telemetry_batch)
    """

    def __init__(self, model_path: str = MODEL_PATH):
        bundle        = joblib.load(model_path)
        self.model    = bundle["model"]
        self.scaler   = bundle["scaler"]
        self.prep     = TelemetryPreprocessor(scaler=self.scaler)

        # Per-node sliding history window for trend detection
        self.history  = defaultdict(lambda: deque(maxlen=TREND_WINDOW))

        self.decision_log      = []
        self.quarantined_nodes = set()
        self._log(f"Agent ready — model loaded from {model_path}")

    # ── STAGE 1: OBSERVE ─────────────────────────────────────
    def observe(self, raw_batch: list) -> list:
        """Receive telemetry batch, drop invalid packets."""
        valid = []
        for pkt in raw_batch:
            ok, errs = self.prep.validate(pkt)
            if ok:
                valid.append(pkt)
            else:
                self._log(f"  WARN  dropping {pkt.get('node_id','?')}: {errs}")
        self._log(f"OBSERVE   {len(valid)}/{len(raw_batch)} packets valid")
        return valid

    # ── STAGE 2: PERCEIVE ────────────────────────────────────
    def perceive(self, valid_packets: list) -> np.ndarray:
        """Normalize packets → (N, 8) feature matrix. Update history."""
        for pkt in valid_packets:
            self.history[pkt["node_id"]].append(
                self.prep.align_packet(pkt)
            )
        matrix, _ = self.prep.batch_to_matrix(valid_packets)
        self._log(f"PERCEIVE  matrix shape {matrix.shape}")
        return matrix

    # ── STAGE 3: REASON ──────────────────────────────────────
    def reason(self, matrix: np.ndarray, valid_packets: list) -> list:
        """Run RF classifier → produce NodeStatus for each node."""
        if matrix.shape[0] == 0:
            return []

        labels = self.model.predict(matrix)
        proba  = self.model.predict_proba(matrix)
        imps   = self.model.feature_importances_
        statuses = []

        for i, pkt in enumerate(valid_packets):
            label = int(labels[i])
            conf  = float(proba[i][label])
            if conf < MIN_CONFIDENCE:
                label = 0          # treat as uncertain → healthy

            breached = self.prep.breached_features(pkt)
            sev      = self._compute_severity(label, conf, breached, imps)
            action   = self._triage_action(label, sev)

            # Trend override: label=0 but sliding window trending bad
            if label == 0 and self._is_trending_bad(pkt.get("node_id", "")):
                action = "watch"
                sev    = max(sev, SEVERITY_WATCH)

            statuses.append(NodeStatus(
                node_id=pkt.get("node_id", "?"),
                label=label, confidence=conf,
                severity=sev, breached=breached, action=action,
                raw_packet=pkt,
            ))

        n_fail = sum(1 for s in statuses if s.label == 1)
        self._log(f"REASON    {len(statuses)-n_fail} healthy · {n_fail} predicted failing")
        return statuses

    # ── STAGE 4: PLAN ────────────────────────────────────────
    def plan(self, statuses: list) -> list:
        """Convert NodeStatus list into concrete AgentDecisions."""
        decisions = []
        fail_frac = sum(1 for s in statuses if s.label == 1) / max(len(statuses), 1)

        for s in statuses:
            # Top 3 most important breached features
            top = sorted(
                s.breached,
                key=lambda f: (
                    self.model.feature_importances_[FEATURES.index(f)]
                    if f in FEATURES else 0
                ),
                reverse=True,
            )[:3]

            if s.label == 1 and s.severity >= SEVERITY_CRITICAL:
                reason = (f"CRITICAL sev={s.severity:.0f} "
                          f"conf={s.confidence:.2f} top={top}")
            elif s.label == 1 and s.severity >= SEVERITY_WARNING:
                reason = f"WARNING sev={s.severity:.0f} top={top}"
            elif s.action == "watch":
                reason = f"WATCH trending bad over {TREND_WINDOW} readings"
            else:
                reason = "HEALTHY"

            decisions.append(AgentDecision(
                node_id=s.node_id, action=s.action,
                reason=reason, severity=s.severity, top_features=top,
            ))

        # Satellite escalation check
        if fail_frac >= SATELLITE_THRESHOLD:
            print(f"  *** SATELLITE ESCALATION: "
                  f"{fail_frac:.0%} nodes failing — activate fallback gateway ***")

        self._log(f"PLAN      {len(decisions)} decisions")
        return decisions

    # ── STAGE 5: ACT ─────────────────────────────────────────
    def act(self, decisions: list) -> dict:
        """Execute decisions. Returns action summary dict."""
        summary = {"alert": [], "reroute": [], "watch": [], "ok": []}
        for d in decisions:
            if d.action in ("alert", "reroute"):
                self.quarantined_nodes.add(d.node_id)
                summary[d.action].append(d.node_id)
                self._log(f"  ACT {d.action.upper():<7} {d.node_id}  "
                          f"sev={d.severity:.0f}  {d.reason}")
            elif d.action == "watch":
                summary["watch"].append(d.node_id)
                self._log(f"  ACT WATCH    {d.node_id}  trending bad")
            else:
                self.quarantined_nodes.discard(d.node_id)
                summary["ok"].append(d.node_id)
        self.decision_log.extend(decisions)
        return summary

    # ── STAGE 6: REFLECT ─────────────────────────────────────
    def reflect(self, next_batch: list) -> dict:
        """
        Compare post-action telemetry against previous decisions.
        Checks whether rerouted nodes recovered.
        """
        outcomes = {}
        rerouted = {
            d.node_id: d
            for d in self.decision_log[-50:]
            if d.action in ("reroute", "alert")
        }
        for pkt in next_batch:
            nid = pkt.get("node_id", "")
            if nid not in rerouted:
                continue
            now  = self.prep.breached_features(pkt)
            prev = rerouted[nid].top_features
            if not now:
                outcomes[nid] = "recovered"
                self.quarantined_nodes.discard(nid)
                self._log(f"  REFLECT {nid} → recovered")
            elif len(now) < len(prev):
                outcomes[nid] = "improved"
                self._log(f"  REFLECT {nid} → improved ({len(now)} breaches left)")
            else:
                outcomes[nid] = "still_degraded"
                self._log(f"  REFLECT {nid} → still degraded")
        return outcomes

    # ── CONVENIENCE WRAPPER ───────────────────────────────────
    def run_cycle(self, raw_batch: list) -> tuple:
        """Run full Observe→Act cycle. Returns (decisions, summary)."""
        valid    = self.observe(raw_batch)
        matrix   = self.perceive(valid)
        statuses = self.reason(matrix, valid)
        decisions= self.plan(statuses)
        summary  = self.act(decisions)
        return decisions, summary

    # ── PRIVATE HELPERS ───────────────────────────────────────
    def _compute_severity(self, label, conf, breached, imps) -> float:
        if label == 0 and not breached:
            return 0.0
        base = conf * 50 if label == 1 else 0.0
        breach_score = sum(
            imps[FEATURES.index(f)]
            for f in breached if f in FEATURES
        ) * 50
        return min(100.0, round(base + breach_score, 1))

    def _triage_action(self, label, severity) -> str:
        if label == 0 and severity < SEVERITY_WATCH: return "ok"
        if severity >= SEVERITY_CRITICAL:            return "alert"
        if severity >= SEVERITY_WARNING:             return "reroute"
        return "watch"

    def _is_trending_bad(self, node_id: str) -> bool:
        hist = list(self.history[node_id])
        if len(hist) < TREND_WINDOW:
            return False
        pkt_trend   = hist[-1].get("pkt_loss_pct", 0)  - hist[0].get("pkt_loss_pct", 0)
        queue_trend = hist[-1].get("queue_len_pkts", 0) - hist[0].get("queue_len_pkts", 0)
        return pkt_trend > 8.0 or queue_trend > 6

    def _log(self, *args):
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] PM-AGENT", *args)
