import json
from collections import defaultdict

from numpy import _1DShapeT

# Load the JSON file
with open(
    "/home/lucas/Documents/soccer_data/statsbomb/open-data/data/events/3825818.json"
) as f:
    events = json.load(f)

edges = defaultdict(int)
player_positions = defaultdict(list)

for ev in events:
    if ev["type"]["name"] == "Pass":
        passer = ev["player"]["id"]
        recipient = ev["pass"].get("recipient", {}).get("id")
        outcome = ev["pass"].get("outcome", {"name": "Complete"})["name"]

        if outcome == "Complete" and recipient is not None:
            # Edge
            edges[(passer, recipient)] += 1

            # Coordinates
            loc_start = ev["location"]
            loc_end = ev["pass"]["end_location"]

            # Assign positions to players
            player_positions[passer].append(loc_start)
            player_positions[recipient].append(loc_end)

# Compute average location for each player
avg_positions = {}
for (
    player_id,
    coords,
) in player_positions.items():
    xs = [pt[0] for pt in coords]
    ys = [pt[1] for pt in coords]
    avg_positions[player_id] = [sum(xs) / len(xs), sum(ys) / len(ys)]

# Build a JSON-friendly structure for export (nodes + links)
nodes = [{"id": pid, "x": pos[0], "y": pos[1]} for pid, pos in avg_positions.items()]
links = [
    {"source": src, "target": tgt, "value": count}
    for (src, tgt), count in edges.items()
]

network = {"nodes": nodes, "links": links}

with open("pass_network.json", "w") as f:
    json.dump(network, f, indent=2)

print("Pass network exported to pass_network.json")
