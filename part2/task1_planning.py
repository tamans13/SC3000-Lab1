from .gridworld import (ACTIONS, GOAL, GAMMA, all_states,
                       transitions, print_value_function,
                       print_policy, compare_policies)

# value iteration
def value_iteration(theta=1e-6):
    """Compute optimal value function V* and policy via Value Iteration."""
    states = all_states()
    V = {s: 0.0 for s in states}

    iteration = 0
    while True:
        delta = 0
        for s in states:
            if s == GOAL:
                V[s] = 0.0
                continue
            action_values = []
            for a in ACTIONS:
                q = sum(p * (r + GAMMA * V[ns]) for p, ns, r in transitions(s, a))
                action_values.append(q)
            best = max(action_values)
            delta = max(delta, abs(best - V[s]))
            V[s] = best
        iteration += 1
        if delta < theta:
            break

    # Extract policy
    policy = {}
    for s in states:
        if s == GOAL:
            policy[s] = None
            continue
        best_a = max(ACTIONS, key=lambda a: sum(p * (r + GAMMA * V[ns])
                                                 for p, ns, r in transitions(s, a)))
        policy[s] = best_a

    print(f"Value Iteration converged in {iteration} iterations.")
    return V, policy

# policy iteration
def policy_evaluation(policy, theta=1e-6):
    """Evaluate a given policy."""
    states = all_states()
    V = {s: 0.0 for s in states}
    while True:
        delta = 0
        for s in states:
            if s == GOAL or policy[s] is None:
                V[s] = 0.0
                continue
            a = policy[s]
            v = sum(p * (r + GAMMA * V[ns]) for p, ns, r in transitions(s, a))
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < theta:
            break
    return V


def policy_iteration():
    """Compute optimal policy via Policy Iteration."""
    states = all_states()
    # Initialize arbitrary policy
    policy = {s: 'Up' for s in states if s != GOAL}
    policy[GOAL] = ''  # Goal state has no action

    iteration = 0
    while True:
        V = policy_evaluation(policy)
        policy_stable = True
        for s in states:
            if s == GOAL:
                continue
            old_a = policy[s]
            best_a = max(ACTIONS, key=lambda a: sum(p * (r + GAMMA * V[ns])
                                                     for p, ns, r in transitions(s, a)))
            policy[s] = best_a
            if old_a != best_a:
                policy_stable = False
        iteration += 1
        if policy_stable:
            break

    print(f"Policy Iteration converged in {iteration} iterations.")
    return V, policy

if __name__ == "__main__":
    V_vi, policy_vi = value_iteration()
    print_value_function(V_vi)
    print_policy(policy_vi)