# Reinforcement Learning & Shortest Path Project

This project implements algorithms for **shortest path planning** and **reinforcement learning** tasks.

The project contains:

- **Part 1:** Shortest path computation using NYC map data
  - Dijkstra's Algorithm (no energy constraint)
  - BFS with energy budget constraint
  - A* Search with energy budget constraint
- **Part 2:** Gridworld reinforcement learning algorithms:
  - Value Iteration
  - Policy Iteration
  - Monte Carlo Learning
  - Q-Learning

All tasks can be run from a single entry point (`main.py`) or individually.

---

# 📁 Project Structure

```
sc3000-lab1/
├── main.py
├── part1/
│   ├── task1.py
│   ├── task2.py
│   └── task3.py
├── part2/
│   ├── gridworld.py
│   ├── task1_planning.py
│   ├── task2_montecarlo.py
│   └── task3_qlearning.py
└── data/
    ├── G.json
    ├── Coord.json
    ├── Dist.json
    └── Cost.json
```

### Folder Descriptions

**main.py**  
Runs the entire project including Part 1 and all Part 2 tasks.

**part1/**  
Contains shortest path implementations.

- `task1.py` → Dijkstra's Algorithm (no energy constraint)
- `task2.py` → BFS with energy budget constraint
- `task3.py` → A* Search with energy budget constraint

**part2/**  
Contains reinforcement learning implementations.

- `gridworld.py` → Gridworld environment definition  
- `task1_planning.py` → Value Iteration and Policy Iteration  
- `task2_montecarlo.py` → Monte Carlo learning  
- `task3_qlearning.py` → Q-Learning implementation  

**data/**  
Contains NYC road network data files.

- `G.json` → Adjacency list of the road network graph
- `Coord.json` → Node coordinates
- `Dist.json` → Edge distances between node pairs
- `Cost.json` → Edge energy costs between node pairs

---

# One-time Setup

Run these commands **once** before running the project.

Open **Terminal** and run:

```bash
# Install required dependency
pip3 install numpy
```
---

# Running the Entire Project

To run **everything (Part 1 + all Part 2 tasks)**:

```bash
python3 main.py
```

This will sequentially execute:

1. NYC shortest path — **Dijkstra** (no energy constraint)
2. NYC shortest path — **BFS** (with energy constraint)
3. NYC shortest path — **A\* Search** (with energy constraint)
4. Gridworld **Value Iteration & Policy Iteration**  
5. **Monte Carlo learning**  
6. **Q-Learning**

# Running Individual Tasks

## Part 1 — Task 1 (Dijkstra's Algorithm)

Runs **Dijkstra's Algorithm** to find the shortest path with no energy constraint.

```bash
python3 -m part1.task1
```

---

## Part 1 — Task 2 (BFS with Energy Constraint)

Runs **Breadth-First Search** to find the shortest path within the energy budget.

```bash
python3 -m part1.task2
```

---

## Part 1 — Task 3 (A* Search with Energy Constraint)

Runs **A* Search** to find the shortest path within the energy budget.

```bash
python3 -m part1.task3
```

---

## Part 2 — Task 1 (Planning Algorithms)

Runs **Value Iteration and Policy Iteration**.

```bash
python3 -m part2.task1_planning
```

---

## Part 2 — Task 2 (Monte Carlo Learning)

```bash
python3 -m part2.task2_montecarlo
```

---

## Part 2 — Task 3 (Q-Learning)

```bash
python3 -m part2.task3_qlearning
```

---
