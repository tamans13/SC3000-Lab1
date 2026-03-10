import numpy as np
import random
from collections import defaultdict
from .gridworld import (ACTIONS, ACTION_IDX, GOAL, GAMMA,
                       all_states, run_episode, print_policy,
                       compare_policies, evaluate_policy_returns)

def monte_carlo_control(num_episodes=50000, epsilon=0.1):
    """MC Control with epsilon-greedy policy."""
    Q = defaultdict(lambda: np.zeros(len(ACTIONS)))
    returns_sum = defaultdict(lambda: np.zeros(len(ACTIONS)))
    returns_count = defaultdict(lambda: np.zeros(len(ACTIONS)))

    def policy_fn(state):
        if random.random() < epsilon:
            return random.choice(ACTIONS)
        return ACTIONS[np.argmax(Q[state])]

    for ep in range(1, num_episodes + 1):
        episode = run_episode(policy_fn)
        G = 0
        visited = set()
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = GAMMA * G + reward
            sa = (state, action)
            if sa not in visited:  # first-visit MC
                visited.add(sa)
                ai = ACTION_IDX[action]
                returns_sum[state][ai] += G
                returns_count[state][ai] += 1
                Q[state][ai] = returns_sum[state][ai] / returns_count[state][ai]

        if ep % 10000 == 0:
            print(f"  MC episode {ep}/{num_episodes} done.")

    # Greedy policy from Q
    mc_policy = {}
    for s in all_states():
        if s == GOAL:
            mc_policy[s] = None
        else:
            mc_policy[s] = ACTIONS[np.argmax(Q[s])]

    return Q, mc_policy

if __name__ == "__main__":
    Q_mc, policy_mc = monte_carlo_control()
    print_policy(policy_mc)