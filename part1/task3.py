"""
A* Search with Energy Budget Constraint
"""

import json
import heapq
import math
import os
import time

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def run():
    with open(os.path.join(BASE, 'data', 'G.json'))     as f: G     = json.load(f)
    with open(os.path.join(BASE, 'data', 'Coord.json')) as f: Coord = json.load(f)
    with open(os.path.join(BASE, 'data', 'Dist.json'))  as f: Dist  = json.load(f)
    with open(os.path.join(BASE, 'data', 'Cost.json'))  as f: Cost  = json.load(f)

    SOURCE        = '1'
    TARGET        = '50'
    ENERGY_BUDGET = 287932

    # ── Heuristic ─────────────────────────────────────────────────
    # Admissible heuristic: straight-line Euclidean distance to TARGET.
    # Coord values are in the same unit system as Dist, so h(n) <= actual dist.
    def h(node):
        x1, y1 = Coord[node]
        x2, y2 = Coord[TARGET]
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    # ── Pareto-label A* ───────────────────────────────────────────
    # Priority queue entries: (f, g, energy, node)
    # pareto[node] = list of (dist, energy) non-dominated labels
    # prev[(node, dist, energy)] = parent state tuple

    pareto         = {SOURCE: [(0, 0)]}
    prev           = {(SOURCE, 0, 0): None}
    pq             = [(h(SOURCE), 0, 0, SOURCE)]
    best_label     = None
    nodes_expanded = 0
    nodes_pushed   = 1      # source is pushed initially

    def dominated(node, nd, ne):
        # A label is dominated if another label has BOTH lower dist AND lower energy
        for (d, e) in pareto.get(node, []):
            if d <= nd and e <= ne:
                return True
        return False

    def add_label(node, nd, ne):
        if dominated(node, nd, ne):
            return False
        # Remove labels that the new one dominates
        pareto[node] = [(d, e) for (d, e) in pareto.get(node, [])
                        if not (nd <= d and ne <= e)]
        pareto[node].append((nd, ne))
        return True

    t0 = time.time()

    while pq:
        f, g, e, u = heapq.heappop(pq)

        # Skip if this label is no longer on the Pareto frontier
        if not any(abs(pd - g) < 1e-9 and pe == e
                   for (pd, pe) in pareto.get(u, [])):
            continue

        nodes_expanded += 1

        if u == TARGET:
            best_label = (g, e)
            break

        for v in G.get(u, []):
            key = f'{u},{v}'
            ng  = g + Dist[key]
            ne  = e + Cost[key]

            # Skip if over budget
            if ne > ENERGY_BUDGET:
                continue

            if add_label(v, ng, ne):
                prev[(v, ng, ne)] = (u, g, e)
                heapq.heappush(pq, (ng + h(v), ng, ne, v))
                nodes_pushed += 1

    elapsed = time.time() - t0

    # ── Reconstruct path ──────────────────────────────────────────
    if best_label is None:
        print("No feasible path found within the energy budget.")
        return None
    else:
        path  = []
        state = (TARGET, best_label[0], best_label[1])
        while state is not None:
            path.append(state[0])
            state = prev.get(state)
        path.reverse()

        # ── Output ────────────────────────────────────────────────
        total_dist = sum(Dist[f'{path[i]},{path[i+1]}'] for i in range(len(path)-1))
        total_cost = sum(Cost[f'{path[i]},{path[i+1]}'] for i in range(len(path)-1))

        print(f"Shortest path: {'->'.join(path)}")
        print(f"Shortest distance: {total_dist}")
        print(f"Total energy cost: {total_cost}")

        return {
            'nodes_expanded': nodes_expanded,
            'nodes_pushed':   nodes_pushed,
            'time':           elapsed
        }

if __name__ == '__main__':
    run()
