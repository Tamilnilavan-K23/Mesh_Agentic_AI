# =============================================================
#  step6_routing_demo.py
#  Shows dynamic routing with one quarantined node.
#  Traffic automatically avoids the bad node via Dijkstra.
#
#  Usage:
#      python step6_routing_demo.py
# =============================================================

from dynamic_routing_agent import DynamicRoutingAgent

print("=" * 60)
print("  Step 6 — Dynamic Routing Agent Demo")
print("=" * 60)

router = DynamicRoutingAgent()

# 5-node topology — N03 is the failing node (quarantined by PM agent)
topo = [
    {"node_id": "N01", "battery_v": 3.9,  "rssi_dbm": -65,  "pkt_loss_pct": 1.2,
     "temp_c": 38, "uptime_hrs": 48,  "hop_count": 2, "queue_len_pkts": 3,  "tx_rate_kbps": 88},
    {"node_id": "N02", "battery_v": 3.8,  "rssi_dbm": -70,  "pkt_loss_pct": 2.1,
     "temp_c": 41, "uptime_hrs": 72,  "hop_count": 3, "queue_len_pkts": 5,  "tx_rate_kbps": 82},
    {"node_id": "N03", "battery_v": 3.1,  "rssi_dbm": -102, "pkt_loss_pct": 38.0,
     "temp_c": 69, "uptime_hrs": 0.1, "hop_count": 8, "queue_len_pkts": 25, "tx_rate_kbps": 7},
    {"node_id": "N04", "battery_v": 3.85, "rssi_dbm": -67,  "pkt_loss_pct": 1.5,
     "temp_c": 39, "uptime_hrs": 96,  "hop_count": 2, "queue_len_pkts": 4,  "tx_rate_kbps": 87},
    {"node_id": "N05", "battery_v": 4.0,  "rssi_dbm": -63,  "pkt_loss_pct": 0.9,
     "temp_c": 36, "uptime_hrs": 120, "hop_count": 1, "queue_len_pkts": 2,  "tx_rate_kbps": 95},
]

quarantined = {"N03"}   # marked failing by PM agent

print(f"\n  Nodes in topology : {[p['node_id'] for p in topo]}")
print(f"  Quarantined nodes : {quarantined}")
print(f"  Destination       : HQ (rescue server)\n")

router.update_topology(topo, quarantined)
router.compute_all_routes("HQ")
router.print_route_table(quarantined)

print("\n  Node scores (lower = better relay candidate):")
for nid, node in sorted(router.nodes.items()):
    status = "QUARANTINED" if not node.available else "active     "
    print(f"    {nid}  score={node.score:.3f}  [{status}]")

print("\n  [Step 6 complete — run step7_orchestrator.py next]")
