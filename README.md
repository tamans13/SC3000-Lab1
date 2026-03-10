# Reinforcement Learning & Shortest Path Project

This project implements algorithms for **shortest path planning** and **reinforcement learning** tasks.

The project contains:

- **Part 1:** Shortest path computation using NYC map data  
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

**part2/**  
Contains reinforcement learning implementations.

- `gridworld.py` → Gridworld environment definition  
- `task1_planning.py` → Value Iteration and Policy Iteration  
- `task2_montecarlo.py` → Monte Carlo learning  
- `task3_qlearning.py` → Q-Learning implementation  

**data/**  
fill in later

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

1. NYC shortest path computation  
2. Gridworld **Value Iteration & Policy Iteration**  
3. **Monte Carlo learning**  
4. **Q-Learning**

# Running Individual Tasks

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
