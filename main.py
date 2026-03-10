# main.py
import sys, os, random
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), 'part2'))

from part2.task1_planning import value_iteration, policy_iteration
from part2.task2_montecarlo import monte_carlo_control
from part2.task3_qlearning import q_learning
from part2.gridworld import (print_value_function, print_policy,
                             compare_policies, evaluate_policy_returns)

# ── Part 1 ──────────────────────────────────────────────
def solve_part1():
    # your NYC shortest path code here
    pass

# ── Part 2 ──────────────────────────────────────────────
def solve_part2():
    random.seed(42)
    np.random.seed(42)

    print("=" * 60)
    print("TASK 1: Value Iteration & Policy Iteration")
    print("=" * 60)

    print("\n--- Value Iteration ---")
    V_vi, policy_vi = value_iteration()
    print_value_function(V_vi, "Value Function (Value Iteration)")
    print_policy(policy_vi, "Optimal Policy (Value Iteration)")

    print("\n--- Policy Iteration ---")
    V_pi, policy_pi = policy_iteration()
    print_value_function(V_pi, "Value Function (Policy Iteration)")
    print_policy(policy_pi, "Optimal Policy (Policy Iteration)")

    compare_policies(policy_vi, policy_pi, "Value Iteration", "Policy Iteration")

    print("\n--- Task 1 Value Function Spot Checks ---")
    for s in [(0,0), (1,0), (3,3), (4,4)]:
        print(f"  V_VI{s} = {V_vi.get(s, 0):.4f} | V_PI{s} = {V_pi.get(s, 0):.4f}")

    print("\n" + "=" * 60)
    print("TASK 2: Monte Carlo Control (ε-greedy, ε=0.1)")
    print("=" * 60)

    print("\nTraining MC agent for 50,000 episodes...")
    Q_mc, policy_mc = monte_carlo_control(num_episodes=50000, epsilon=0.1)
    print_policy(policy_mc, "Learned Policy (Monte Carlo)")

    compare_policies(policy_vi, policy_mc, "Optimal (VI)", "Monte Carlo")

    avg_mc = evaluate_policy_returns(policy_mc)
    avg_vi = evaluate_policy_returns(policy_vi)
    print(f"\n  Average discounted return (VI policy):  {avg_vi:.4f}")
    print(f"  Average discounted return (MC policy):  {avg_mc:.4f}")

    print("\n" + "=" * 60)
    print("TASK 3: Q-Learning (ε-greedy, ε=0.1, α=0.1)")
    print("=" * 60)

    print("\nTraining Q-Learning agent for 50,000 episodes...")
    Q_ql, policy_ql = q_learning(num_episodes=50000, epsilon=0.1, alpha=0.1)
    print_policy(policy_ql, "Learned Policy (Q-Learning)")

    compare_policies(policy_vi, policy_ql, "Optimal (VI)", "Q-Learning")
    compare_policies(policy_mc, policy_ql, "Monte Carlo", "Q-Learning")

    avg_ql = evaluate_policy_returns(policy_ql)
    print(f"\n  Average discounted return (VI policy):  {avg_vi:.4f}")
    print(f"  Average discounted return (MC policy):  {avg_mc:.4f}")
    print(f"  Average discounted return (QL policy):  {avg_ql:.4f}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
  Task 1 (Value Iteration vs Policy Iteration):
    Both algorithms compute the same optimal policy from the full
    known MDP model. Value Iteration updates V(s) directly;
    Policy Iteration alternates evaluation and improvement steps.

  Task 2 (Monte Carlo):
    MC learns from complete episode returns with no model knowledge.
    It converges more slowly than model-based methods but uses
    ε-greedy exploration to balance exploitation and exploration.

  Task 3 (Q-Learning):
    Q-Learning is an off-policy TD method that updates Q-values
    after every step (not episode). It converges faster than MC
    because it bootstraps from the next state's Q-value immediately,
    making better use of every interaction with the environment.
    """)

if __name__ == "__main__":
    solve_part1()
    solve_part2()