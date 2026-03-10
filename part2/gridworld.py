import numpy as np
import random
from collections import defaultdict

GRID_SIZE = 5
START = (0, 0)
GOAL = (4, 4)
ROADBLOCKS = {(1, 2), (3, 2)}

ACTIONS = ['Up', 'Down', 'Left', 'Right']
ACTION_IDX = {a: i for i, a in enumerate(ACTIONS)}

# Perpendicular actions for stochastic transitions
PERP = {
    'Up':    ['Left', 'Right'],
    'Down':  ['Right', 'Left'],
    'Left':  ['Down', 'Up'],
    'Right': ['Up', 'Down'],
}

DELTA = {
    'Up':    (0, 1),
    'Down':  (0, -1),
    'Left':  (-1, 0),
    'Right': (1, 0),
}

GAMMA = 0.9          # discount factor
STEP_REWARD = -1     # reward per step
GOAL_REWARD = 10     # reward on reaching goal


def all_states():
    """Return all non-roadblock states."""
    return [(x, y) for x in range(GRID_SIZE)
            for y in range(GRID_SIZE)
            if (x, y) not in ROADBLOCKS]


def move(state, action):
    """Deterministic move: returns next state (clamped to grid, blocked by roadblocks)."""
    x, y = state
    dx, dy = DELTA[action]
    nx, ny = x + dx, y + dy
    if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and (nx, ny) not in ROADBLOCKS:
        return (nx, ny)
    return state  # stay


def transitions(state, action):
    """
    Returns list of (prob, next_state, reward).
    Stochastic: 0.8 intended, 0.1 each perpendicular.
    """
    if state == GOAL:
        return [(1.0, GOAL, 0)]  # absorbing

    outcomes = []
    intended_next = move(state, action)
    perp1, perp2 = PERP[action]
    perp1_next = move(state, perp1)
    perp2_next = move(state, perp2)

    for prob, next_s in [(0.8, intended_next), (0.1, perp1_next), (0.1, perp2_next)]:
        reward = GOAL_REWARD if next_s == GOAL else STEP_REWARD
        outcomes.append((prob, next_s, reward))

    # Merge duplicate next states
    merged = defaultdict(float)
    reward_map = {}
    for p, s, r in outcomes:
        merged[s] += p
        reward_map[s] = r  # reward depends only on next state
    return [(p, s, reward_map[s]) for s, p in merged.items()]

def env_step(state, action):
    """Sample next state and reward from the stochastic environment."""
    if state == GOAL:
        return GOAL, 0, True

    roll = random.random()
    perp1, perp2 = PERP[action]
    if roll < 0.8:
        chosen = action
    elif roll < 0.9:
        chosen = perp1
    else:
        chosen = perp2

    next_s = move(state, chosen)
    if next_s == GOAL:
        return GOAL, GOAL_REWARD, True
    return next_s, STEP_REWARD, False


def run_episode(policy_fn, max_steps=500):
    """Run one episode using policy_fn(state) -> action. Returns trajectory."""
    state = START
    episode = []
    for _ in range(max_steps):
        action = policy_fn(state)
        next_s, reward, done = env_step(state, action)
        episode.append((state, action, reward))
        state = next_s
        if done:
            break
    return episode

def print_value_function(V, title="Value Function"):
    print(f"\n{title}:")
    print("     x=0     x=1     x=2     x=3     x=4")
    for y in range(GRID_SIZE - 1, -1, -1):
        row = f"y={y} "
        for x in range(GRID_SIZE):
            s = (x, y)
            if s in ROADBLOCKS:
                row += " [BLOCK]"
            elif s == GOAL:
                row += "  [GOAL]"
            else:
                row += f"  {V.get(s, 0):6.2f}"
        print(row)


def print_policy(policy, title="Policy"):
    symbols = {'Up': '↑', 'Down': '↓', 'Left': '←', 'Right': '→', None: 'G'}
    print(f"\n{title}:")
    print("    x=0  x=1  x=2  x=3  x=4")
    for y in range(GRID_SIZE - 1, -1, -1):
        row = f"y={y} "
        for x in range(GRID_SIZE):
            s = (x, y)
            if s in ROADBLOCKS:
                row += "  [B]"
            else:
                row += f"   {symbols.get(policy.get(s), '?')}"
        print(row)


def compare_policies(p1, p2, name1="Policy 1", name2="Policy 2"):
    print(f"\nPolicy Comparison ({name1} vs {name2}):")
    states = all_states()
    diff = [(s, p1[s], p2[s]) for s in states if s != GOAL and p1.get(s) != p2.get(s)]
    if not diff:
        print("  Policies are identical!")
    else:
        print(f"  {len(diff)} states differ:")
        for s, a1, a2 in diff:
            print(f"    State {s}: {name1}={a1}, {name2}={a2}")


def evaluate_policy_returns(policy, num_episodes=5000):
    """Monte Carlo estimate of average return under a (deterministic) policy."""
    total = 0
    def policy_fn(s):
        return policy.get(s, random.choice(ACTIONS)) or random.choice(ACTIONS)
    for _ in range(num_episodes):
        episode = run_episode(policy_fn)
        G = sum(r * (GAMMA ** t) for t, (_, _, r) in enumerate(episode))
        total += G
    return total / num_episodes