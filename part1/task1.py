"""
SC3000 Lab Assignment 1 – Part 1, Task 1
Dijkstra's Algorithm (no energy constraint)

Finds the shortest distance path from SOURCE to TARGET,
ignoring energy cost entirely.

Source: '1'   Target: '50'
"""

import json
import heapq
import math
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def run():
    with open(os.path.join(BASE, 'data', 'G.json'))    as f: G    = json.load(f)
    with open(os.path.join(BASE, 'data', 'Dist.json')) as f: Dist = json.load(f)
    with open(os.path.join(BASE, 'data', 'Cost.json')) as f: Cost = json.load(f)

    SOURCE = '1'
    TARGET = '50'

    # ── Dijkstra ──────────────────────────────────────────────────
    # Priority queue entries: (cumulative_dist, node)
    # dist[node] = best known distance to reach node so far
    # prev[node] = parent node on the best path

    INF  = math.inf
    dist = {SOURCE: 0}
    prev = {SOURCE: None}
    pq   = [(0, SOURCE)]

    while pq:
        d, u = heapq.heappop(pq)

        if u == TARGET:
            break

        # Skip outdated queue entries
        if d > dist.get(u, INF):
            continue

        for v in G.get(u, []):
            nd = d + Dist[f'{u},{v}']
            if nd < dist.get(v, INF):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))

    # ── Reconstruct path ──────────────────────────────────────────
    path, node = [], TARGET
    while node is not None:
        path.append(node)
        node = prev[node]
    path.reverse()

    # ── Output ────────────────────────────────────────────────────
    total_dist = sum(Dist[f'{path[i]},{path[i+1]}'] for i in range(len(path)-1))
    total_cost = sum(Cost[f'{path[i]},{path[i+1]}'] for i in range(len(path)-1))

    print(f"Shortest path: {'->'.join(path)}")
    print(f"Shortest distance: {total_dist}")
    print(f"Total energy cost: {total_cost}")

if __name__ == '__main__':
    run()
