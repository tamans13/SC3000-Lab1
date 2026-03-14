import numpy as np
import random
from collections import defaultdict
from .gridworld import (
    ACTIONS, ACTION_IDX, GOAL, GAMMA,
    all_states, run_episode, print_policy
)

# --------------------------------------------------
# Monte Carlo Prediction for V(s)
# --------------------------------------------------
def mc_prediction_v(policy, num_episodes=10000):
    """
    First-visit Monte Carlo prediction for state-value function V(s)
    under a fixed deterministic policy.

    policy: dict mapping state -> action
    returns: V dictionary
    """
    returns_sum = defaultdict(float)
    returns_count = defaultdict(int)
    V = {s: 0.0 for s in all_states()}

    def policy_fn(state):
        return policy[state]

    for ep in range(1, num_episodes + 1):
        episode = run_episode(policy_fn)

        # Compute returns for each timestep
        returns = []
        G = 0
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = GAMMA * G + reward
            returns.append((state, G))
        returns.reverse()

        # First-visit MC update
        visited_states = set()
        for state, G in returns:
            if state not in visited_states:
                visited_states.add(state)
                returns_sum[state] += G
                returns_count[state] += 1
                V[state] = returns_sum[state] / returns_count[state]

        if ep % 2000 == 0:
            print(f"  MC Prediction(V) episode {ep}/{num_episodes} done.")

    V[GOAL] = 0.0
    return V


# --------------------------------------------------
# Monte Carlo Prediction for Q(s,a)
# --------------------------------------------------
def mc_prediction_q(policy, num_episodes=10000):
    """
    First-visit Monte Carlo prediction for action-value function Q(s,a)
    under a fixed deterministic policy.

    policy: dict mapping state -> action
    returns: Q dictionary
    """
    returns_sum = defaultdict(lambda: np.zeros(len(ACTIONS)))
    returns_count = defaultdict(lambda: np.zeros(len(ACTIONS)))
    Q = defaultdict(lambda: np.zeros(len(ACTIONS)))

    def policy_fn(state):
        return policy[state]

    for ep in range(1, num_episodes + 1):
        episode = run_episode(policy_fn)

        # Compute returns for each timestep
        returns = []
        G = 0
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = GAMMA * G + reward
            returns.append((state, action, G))
        returns.reverse()

        # First-visit MC update
        visited_sa = set()
        for state, action, G in returns:
            sa = (state, action)
            if sa not in visited_sa:
                visited_sa.add(sa)
                ai = ACTION_IDX[action]
                returns_sum[state][ai] += G
                returns_count[state][ai] += 1
                Q[state][ai] = returns_sum[state][ai] / returns_count[state][ai]

        if ep % 2000 == 0:
            print(f"  MC Prediction(Q) episode {ep}/{num_episodes} done.")

    return Q


# --------------------------------------------------
# Monte Carlo Control
# --------------------------------------------------
def monte_carlo_control(num_episodes=50000, epsilon=0.1):
    """
    First-visit MC control with epsilon-greedy policy improvement.
    Learns Q(s,a) and returns greedy policy.
    """
    Q = defaultdict(lambda: np.zeros(len(ACTIONS)))
    returns_sum = defaultdict(lambda: np.zeros(len(ACTIONS)))
    returns_count = defaultdict(lambda: np.zeros(len(ACTIONS)))

    def policy_fn(state):
        if state == GOAL:
            return None
        if random.random() < epsilon:
            return random.choice(ACTIONS)
        return ACTIONS[np.argmax(Q[state])]

    for ep in range(1, num_episodes + 1):
        episode = run_episode(policy_fn)

        # Compute returns first
        returns = []
        G = 0
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = GAMMA * G + reward
            returns.append((state, action, G))
        returns.reverse()

        # First-visit MC update
        visited_sa = set()
        for state, action, G in returns:
            sa = (state, action)
            if sa not in visited_sa:
                visited_sa.add(sa)
                ai = ACTION_IDX[action]
                returns_sum[state][ai] += G
                returns_count[state][ai] += 1
                Q[state][ai] = returns_sum[state][ai] / returns_count[state][ai]

        if ep % 10000 == 0:
            print(f"  MC Control episode {ep}/{num_episodes} done.")

    # Greedy policy from Q
    mc_policy = {}
    for s in all_states():
        if s == GOAL:
            mc_policy[s] = None
        else:
            mc_policy[s] = ACTIONS[np.argmax(Q[s])]

    return Q, mc_policy


if __name__ == "__main__":
    # Example fixed policy: always move Right unless at goal
    test_policy = {}
    for s in all_states():
        if s == GOAL:
            test_policy[s] = None
        else:
            test_policy[s] = "Right"

    V_mc = mc_prediction_v(test_policy, num_episodes=5000)
    print("\nMC Prediction V(s):")
    for s in sorted(V_mc):
        print(s, round(V_mc[s], 3))

    Q_mc, policy_mc = monte_carlo_control()
    print_policy(policy_mc, "Learned Policy (Monte Carlo Control)")