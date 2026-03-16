# main.py
import sys, os, random
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), 'part2'))

from part2.task1_planning import value_iteration, policy_iteration
from part2.task2_montecarlo import monte_carlo_control, mc_prediction_v
from part2.task3_qlearning import q_learning
from part2.gridworld import (
    print_value_function,
    print_policy,
    compare_policies,
    evaluate_policy_returns
)

# ── Part 1 ──────────────────────────────────────────────
def solve_part1():
    from part1 import task1, task2, task3

    print("PART 1: SHORTEST PATH WITH AN ENERGY BUDGET")

    print("\n" + "=" * 60)
    print("TASK 1: Dijkstra (No Energy Constraint)")
    print("=" * 60)
    s1 = task1.run()

    print("\n" + "=" * 60)
    print("TASK 2: BFS (With Energy Constraint)")
    print("=" * 60)
    s2 = task2.run()

    print("\n" + "=" * 60)
    print("TASK 3: A* (With Energy Constraint)")
    print("=" * 60)
    s3 = task3.run()

    print("\n" + "=" * 60)
    print("COMPARATIVE ANALYSIS")
    print("=" * 60)
    print(f"\n  {'Algorithm':<20} {'Nodes Expanded':>15} {'Nodes Pushed':>13} {'Time (s)':>10}")
    print("  " + "-" * 60)
    print(f"  {'Task 1 Dijkstra':<20} {s1['nodes_expanded']:>15,} {s1['nodes_pushed']:>13,} {s1['time']:>10.4f}")
    print(f"  {'Task 2 BFS':<20} {s2['nodes_expanded']:>15,} {s2['nodes_pushed']:>13,} {s2['time']:>10.4f}")
    print(f"  {'Task 3 A*':<20} {s3['nodes_expanded']:>15,} {s3['nodes_pushed']:>13,} {s3['time']:>10.4f}")
    print(f"""
  A* explored the fewest nodes ({s3['nodes_expanded']:,}), compared to BFS ({s2['nodes_expanded']:,})
  and Dijkstra ({s1['nodes_expanded']:,}), demonstrating the effectiveness of the
  Euclidean heuristic in guiding the search toward the target.

  BFS explored the most nodes as it searches level by level
  without any directional guidance, making it the slowest.

  Dijkstra is faster than A* in terms of time as it solves a
  simpler problem with no energy constraint, resulting in less
  computational overhead despite expanding more nodes than A*.
    """)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print("""
  Task 1 (Dijkstra):
    Finds the globally optimal shortest path from node 1 to node 50
    using a min-heap priority queue. No energy constraint applied.

  Task 2 (BFS):
    Finds a shortest path within the energy budget of 287,932 using
    a FIFO queue. Pruning via best_energy dictionary prevents
    redundant re-exploration of nodes via costlier energy routes.

  Task 3 (A*):
    Finds the shortest path within the energy budget of 287,932 using
    an admissible Euclidean heuristic. The Pareto-label approach
    tracks multiple (distance, energy) labels per node to avoid
    discarding paths that trade off between distance and energy.
    """)

# ── Part 2 ──────────────────────────────────────────────
def solve_part2():
    
    print("PART 2: SOLVING MDP AND REINFORCEMENT LEARNING PROBLEMS USING A GRID WORLD")
    
    random.seed(42)
    np.random.seed(42)

    print("\n" + "=" * 60)
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
    for s in [(0, 0), (1, 0), (3, 3), (4, 4)]:
        print(f"  V_VI{s} = {V_vi.get(s, 0):.4f} | V_PI{s} = {V_pi.get(s, 0):.4f}")

    print("\n" + "=" * 60)
    print("TASK 2A: Monte Carlo Prediction")
    print("=" * 60)

    print("\nEstimating V(s) for the Value Iteration policy using MC prediction...")
    V_mc_pred = mc_prediction_v(policy_vi, num_episodes=10000)
    print_value_function(V_mc_pred, "MC Predicted Value Function under VI Policy")

    print("\n--- Task 2A Value Function Spot Checks ---")
    for s in [(0, 0), (1, 0), (3, 3), (4, 4)]:
        print(f"  V_MC{s} = {V_mc_pred.get(s, 0):.4f} | V_VI{s} = {V_vi.get(s, 0):.4f}")

    print("\n" + "=" * 60)
    print("TASK 2B: Monte Carlo Control (ε-greedy, ε=0.1)")
    print("=" * 60)

    print("\nTraining MC agent for 50,000 episodes...")
    Q_mc, policy_mc = monte_carlo_control(num_episodes=50000, epsilon=0.1)
    print_policy(policy_mc, "Learned Policy (Monte Carlo Control)")

    compare_policies(policy_vi, policy_mc, "Optimal (VI)", "Monte Carlo Control")

    avg_vi = evaluate_policy_returns(policy_vi)
    avg_mc = evaluate_policy_returns(policy_mc)
    print(f"\n  Average discounted return (VI policy):  {avg_vi:.4f}")
    print(f"  Average discounted return (MC policy):  {avg_mc:.4f}")

    print("\n" + "=" * 60)
    print("TASK 3: Q-Learning (ε-greedy, ε=0.1, α=0.1)")
    print("=" * 60)

    print("\nTraining Q-Learning agent for 50,000 episodes...")
    Q_ql, policy_ql = q_learning(num_episodes=50000, epsilon=0.1, alpha=0.1)
    print_policy(policy_ql, "Learned Policy (Q-Learning)")

    compare_policies(policy_vi, policy_ql, "Optimal (VI)", "Q-Learning")
    compare_policies(policy_mc, policy_ql, "Monte Carlo Control", "Q-Learning")

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

  Task 2A (Monte Carlo Prediction):
    MC prediction estimates the value function of a fixed policy
    using complete sampled episodes. Here, it is used to estimate
    V(s) under the Value Iteration policy and compare it against
    the model-based value function.

  Task 2B (Monte Carlo Control):
    MC control learns from complete episode returns with no model
    knowledge. It converges more slowly than model-based methods
    but uses ε-greedy exploration to balance exploitation and
    exploration while improving the policy.

  Task 3 (Q-Learning):
    Q-Learning is an off-policy TD method that updates Q-values
    after every step (not episode). It converges faster than MC
    because it bootstraps from the next state's Q-value immediately,
    making better use of every interaction with the environment.
    """)

if __name__ == "__main__":
    solve_part1()
    solve_part2()
