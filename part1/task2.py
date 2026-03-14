"""
SC3000 Lab Assignment 1 – Part 1, Task 2
Breadth-First Search (BFS) with Energy Budget Constraint

BFS explores nodes level by level using a FIFO queue.
It finds the path with the fewest hops that stays within
the energy budget.

Pruning rule: for each node, we only enqueue it if we are
arriving with LESS energy than any previous visit. If we
have already reached a node with lower energy, there is no
reason to explore it again via a more energy-costly route.

Source: '1'   Target: '50'   Energy Budget: 287932
"""

import json
import os
from collections import deque

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def run():
    with open(os.path.join(BASE, 'data', 'G.json'))    as f: G    = json.load(f)
    with open(os.path.join(BASE, 'data', 'Dist.json')) as f: Dist = json.load(f)
    with open(os.path.join(BASE, 'data', 'Cost.json')) as f: Cost = json.load(f)

    SOURCE        = '1'
    TARGET        = '50'
    ENERGY_BUDGET = 287932

    # ── BFS ───────────────────────────────────────────────────────
    # Queue entries: (node, energy_used)
    # best_energy[node] = minimum energy used to reach node so far
    # prev[node]        = parent node on the discovered path

    queue       = deque()
    best_energy = {SOURCE: 0}
    prev        = {SOURCE: None}
    found       = False

    queue.append((SOURCE, 0))

    while queue:
        node, energy = queue.popleft()    # FIFO -> breadth-first

        if node == TARGET:
            found = True
            break

        for v in G.get(node, []):
            new_energy = energy + Cost[f'{node},{v}']

            # Skip if over budget
            if new_energy > ENERGY_BUDGET:
                continue

            # Only enqueue if arriving with better (lower) energy than before
            if new_energy < best_energy.get(v, float('inf')):
                best_energy[v] = new_energy
                prev[v]        = node
                queue.append((v, new_energy))

    # ── Reconstruct path ──────────────────────────────────────────
    if not found:
        print("No feasible path found within the energy budget.")
    else:
        path, node = [], TARGET
        while node is not None:
            path.append(node)
            node = prev[node]
        path.reverse()

        # ── Output ────────────────────────────────────────────────
        total_dist = sum(Dist[f'{path[i]},{path[i+1]}'] for i in range(len(path)-1))
        total_cost = sum(Cost[f'{path[i]},{path[i+1]}'] for i in range(len(path)-1))

        print(f"Shortest path: {'->'.join(path)}")
        print(f"Shortest distance: {total_dist}")
        print(f"Total energy cost: {total_cost}")

if __name__ == '__main__':
    run()
