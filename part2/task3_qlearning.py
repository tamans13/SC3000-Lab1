import numpy as np
import random
from collections import defaultdict
from .gridworld import (ACTIONS, ACTION_IDX, GOAL, GAMMA, START,
                       all_states, env_step, print_policy,
                       compare_policies, evaluate_policy_returns)

def q_learning(num_episodes=50000, epsilon=0.1, alpha=0.1):
    """Tabular Q-Learning with epsilon-greedy exploration."""
    Q = defaultdict(lambda: np.zeros(len(ACTIONS)))

    for ep in range(1, num_episodes + 1):
        state = START
        for _ in range(500):
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = random.choice(ACTIONS)
            else:
                action = ACTIONS[np.argmax(Q[state])]

            next_s, reward, done = env_step(state, action)
            ai = ACTION_IDX[action]

            # Q-learning update
            best_next = np.max(Q[next_s]) if not done else 0.0
            Q[state][ai] += alpha * (reward + GAMMA * best_next - Q[state][ai])

            state = next_s
            if done:
                break

        if ep % 10000 == 0:
            print(f"  Q-Learning episode {ep}/{num_episodes} done.")

    # Greedy policy from Q
    ql_policy = {}
    for s in all_states():
        if s == GOAL:
            ql_policy[s] = None
        else:
            ql_policy[s] = ACTIONS[np.argmax(Q[s])]

    return Q, ql_policy

if __name__ == "__main__":
    Q_ql, policy_ql = q_learning()
    print_policy(policy_ql)