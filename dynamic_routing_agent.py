# =============================================================
#  dynamic_routing_agent.py
#  Dijkstra-based mesh routing — avoids quarantined nodes
#  set by the Predictive Maintenance Agent
# =============================================================

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from config import FEATURES, FEATURE_RANGES, ROUTING_WEIGHTS
from preprocessor import TelemetryPreprocessor


# ----------------------------------------------------------
#  Data classes
# ----------------------------------------------------------

@dataclass
class MeshNode:
    node_id:   str
    telemetry: dict
    score:     float = 0.0    # routing cost (lower = better relay)
    available: bool  = True   # False when quarantined


@dataclass
class Route:
    path:                 list   # ordered node_ids source → HQ
    total_score:          float
    hop_count:            int
    estimated_latency_ms: float
    confidence:           str    # "high" | "medium" | "low"
    timestamp:            datetime = field(default_factory=datetime.now)


# ----------------------------------------------------------
#  Agent
# ----------------------------------------------------------

class DynamicRoutingAgent:
    """
    Maintains a live mesh topology and computes the lowest-cost
    path from any node to rescue HQ using Dijkstra.

    Works alongside PredictiveMaintenanceAgent:
        router.update_topology(batch, pm_agent.quarantined_nodes)
        route = router.find_best_route("N07")
    """

    def __init__(self):
        self.nodes       = {}    # node_id → MeshNode
        self.adjacency   = {}    # node_id → [neighbour_ids]
        self.route_table = {}    # node_id → best Route to HQ
        self._log("Router initialized")

    # ── Topology update ───────────────────────────────────────
    def update_topology(self, telemetry_batch: list, quarantined_nodes: set):
        """
        Refresh node scores and availability from latest telemetry.
        Called every cycle after the PM agent acts.
        """
        for pkt in telemetry_batch:
            pkt  = TelemetryPreprocessor.align_packet(pkt)
            nid  = pkt.get("node_id", "?")
            self.nodes[nid] = MeshNode(
                node_id=nid,
                telemetry=pkt,
                score=self._compute_node_score(pkt),
                available=(nid not in quarantined_nodes),
            )
        self._rebuild_adjacency()
        avail = sum(1 for n in self.nodes.values() if n.available)
        self._log(f"Topology: {len(self.nodes)} nodes | "
                  f"{avail} available | "
                  f"{len(quarantined_nodes)} quarantined")

    # ── Route computation ─────────────────────────────────────
    def find_best_route(self, source_id: str, destination: str = "HQ") -> Optional[Route]:
        """
        Dijkstra over the mesh graph.
        Quarantined / offline nodes have infinite edge weight.
        """
        if source_id not in self.nodes:
            return None

        all_ids   = list(self.nodes.keys()) + [destination]
        INF       = float("inf")
        dist      = {n: INF  for n in all_ids}
        prev      = {n: None for n in all_ids}
        dist[source_id] = 0.0
        unvisited = set(all_ids)

        while unvisited:
            u = min(unvisited, key=lambda n: dist[n])
            if dist[u] == INF:
                break
            unvisited.remove(u)
            for v in self._get_neighbours(u, destination):
                if v not in unvisited:
                    continue
                w   = 0 if v == destination else self._edge_weight(v)
                alt = dist[u] + w
                if alt < dist[v]:
                    dist[v] = alt
                    prev[v] = u

        if dist[destination] == INF:
            self._log(f"  No path from {source_id} to {destination}")
            return None

        # Reconstruct path
        path, cur = [], destination
        while cur is not None:
            path.append(cur)
            cur = prev[cur]
        path.reverse()

        route = Route(
            path=path,
            total_score=round(dist[destination], 4),
            hop_count=len(path) - 1,
            estimated_latency_ms=self._estimate_latency(path),
            confidence=self._route_confidence(dist[destination], len(path)),
        )
        self.route_table[source_id] = route
        return route

    def compute_all_routes(self, destination: str = "HQ") -> dict:
        """Compute best route for every node in the topology."""
        for nid in list(self.nodes.keys()):
            self.find_best_route(nid, destination)
        return self.route_table

    def get_route_table(self) -> list:
        """Return routing table as a list of dicts (sorted best → worst)."""
        return sorted([
            {
                "node_id":    nid,
                "path":       " → ".join(r.path),
                "hops":       r.hop_count,
                "score":      r.total_score,
                "latency_ms": r.estimated_latency_ms,
                "confidence": r.confidence,
                "available":  self.nodes.get(nid, MeshNode("?", {})).available,
            }
            for nid, r in self.route_table.items()
        ], key=lambda x: (0 if x["available"] else 1, x["score"]))

    def print_route_table(self, quarantined: set = None):
        """Pretty-print the current routing table to stdout."""
        quarantined = quarantined or set()
        table = self.get_route_table()
        print(f"\n  {'NODE':<6} {'PATH':<36} {'HOPS':<5} "
              f"{'SCORE':<7} {'LATENCY':>9}ms  {'CONF':<8} STATUS")
        print("  " + "─" * 78)
        for r in table:
            status = "[QUARANTINED]" if r["node_id"] in quarantined else "[ACTIVE]     "
            flag   = "!" if r["node_id"] in quarantined else " "
            print(f" {flag} {r['node_id']:<6} {r['path']:<36} {r['hops']:<5} "
                  f"{r['score']:<7.3f} {r['latency_ms']:>10.0f}   "
                  f"{r['confidence']:<8} {status}")

    # ── Private helpers ───────────────────────────────────────
    def _compute_node_score(self, pkt: dict) -> float:
        """Weighted routing cost — lower is a better relay choice."""
        score = 0.0
        for feat, weight in ROUTING_WEIGHTS.items():
            val      = pkt.get(feat, 0)
            lo, hi   = FEATURE_RANGES[feat]
            norm     = (val - lo) / (hi - lo + 1e-9)
            # Invert features where lower value = worse node
            if feat in ("rssi_dbm", "battery_v", "tx_rate_kbps"):
                norm = 1.0 - norm
            score += weight * max(0.0, min(1.0, norm))
        return round(score, 4)

    def _rebuild_adjacency(self):
        """Two nodes are adjacent if both available and have RSSI > -105 dBm."""
        nids = list(self.nodes.keys())
        self.adjacency = {nid: [] for nid in nids}
        for i, a in enumerate(nids):
            for b in nids[i + 1:]:
                na, nb = self.nodes[a], self.nodes[b]
                if na.available and nb.available:
                    rssi_a = na.telemetry.get("rssi_dbm", -120)
                    rssi_b = nb.telemetry.get("rssi_dbm", -120)
                    if rssi_a > -105 and rssi_b > -105:
                        self.adjacency[a].append(b)
                        self.adjacency[b].append(a)

    def _get_neighbours(self, node_id: str, destination: str) -> list:
        if node_id == destination:
            return []
        nb = list(self.adjacency.get(node_id, []))
        if node_id in self.nodes and self.nodes[node_id].available:
            nb.append(destination)
        return nb

    def _edge_weight(self, node_id: str) -> float:
        if node_id not in self.nodes:
            return float("inf")
        n = self.nodes[node_id]
        return float("inf") if not n.available else n.score

    def _estimate_latency(self, path: list) -> float:
        """50ms base per hop + 5ms per queued packet at each relay."""
        total = 0.0
        for nid in path[:-1]:      # exclude final destination
            if nid in self.nodes:
                q = self.nodes[nid].telemetry.get("queue_len_pkts", 0)
                total += 50.0 + q * 5.0
        return round(total, 1)

    def _route_confidence(self, score: float, hops: int) -> str:
        if score < 0.3 and hops <= 3: return "high"
        if score < 0.6 and hops <= 5: return "medium"
        return "low"

    @staticmethod
    def _normalize_keys(pkt: dict) -> dict:
        """Lightweight key normalization (fallback if preprocessor unavailable)."""
        from config import COLUMN_RENAME_MAP
        out = {}
        for k, v in pkt.items():
            key = k.strip().lower().replace(" ", "_").replace("%", "pct")
            out[COLUMN_RENAME_MAP.get(key, key)] = v
        return out

    def _log(self, *args):
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] ROUTER  ", *args)