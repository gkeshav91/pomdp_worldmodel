import numpy as np
from mdp import ModelBasedAgent, cMDP
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt


def algorithm_2_extract_transition(agent: ModelBasedAgent, env: cMDP,
                                   source_state: int, action: int,
                                   target_state: int, n_trials: int = 50,
                                   alternative_action: int = None) -> float:
    """
    Algorithm 2: Extract transition probability using binary search with BINOMIAL CDF.

    This implements the switch-goal approach from Richens et al.:
    - Query agent with paired goals ψ_a(k,n) ∨ ψ_b(k,n)
    - ψ_a: succeed ≤k times out of n trials
    - ψ_b: succeed >k times out of n trials
    - Find k* where agent switches preference
    - Estimate P(target_state | source_state, action) ≈ k*/n
    """
    if alternative_action is None:
        # Pick different action
        alternative_action = (action + 1) % env.n_actions

    # Find switch point k* using linear search
    # (could be optimized to binary search)
    k_star = n_trials  # Default if agent always prefers goal_a

    for k in range(1, n_trials + 1):
        chosen_action = agent.optimal_action_for_goal(
            state=source_state,
            goal_a=action,
            goal_b=alternative_action,
            target_state=source_state,  # Agent needs to return here for trials
            success_state=target_state,  # This is what we're measuring
            n_trials=n_trials,
            threshold=k
        )

        # Agent switches from goal_b (>k successes) to goal_a (≤k successes)
        # when k crosses the median of Binomial(n, p)
        if chosen_action == action:
            k_star = k
            break

    # Estimate probability with continuity correction
    # The -0.5 accounts for discretization bias (see Richens Theorem 1)
    p_hat = (k_star - 0.5) / n_trials
    return np.clip(p_hat, 0, 1)


def compute_metrics(agent: ModelBasedAgent, env: cMDP,
                   n_trials: int = 50) -> Tuple[float, float, float]:
    """
    Compute average error and regret metrics.

    Returns:
        avg_error: Average |P_hat(s'|s,a) - P(s'|s,a)|
        avg_regret: Average regret (simplified)
        max_depth: Maximum depth agent can handle with low regret
    """
    errors = []

    # Sample subset of transitions to test
    n_test_samples = min(100, env.n_states * env.n_actions)

    for _ in range(n_test_samples):
        s = np.random.randint(env.n_states)
        a = np.random.randint(env.n_actions)
        s_prime = np.random.randint(env.n_states)

        # True probability
        p_true = env.get_transition_prob(s, a, s_prime)

        # Skip very low probability transitions (as in paper)
        if p_true < 0.01:
            continue

        # Extracted probability
        p_hat = algorithm_2_extract_transition(agent, env, s, a, s_prime, n_trials)

        error = abs(p_hat - p_true)
        errors.append(error)

    avg_error = np.mean(errors) if errors else 0.0

    # Compute regret (simplified: compare learned vs true model)
    regrets = []
    for _ in range(50):
        s = np.random.randint(env.n_states)
        a = np.random.randint(env.n_actions)

        # Compare agent's model to true model
        agent_dist = agent.transitions[a, s, :]
        true_dist = env.transitions[a, s, :]

        regret = np.sum(np.abs(agent_dist - true_dist)) / 2  # Total variation
        regrets.append(regret)

    avg_regret = np.mean(regrets)

    # Approximate max depth (inverse of regret)
    max_depth = 1.0 / (avg_regret + 1e-6)

    return avg_error, avg_regret, max_depth


def run_single_experiment(n_samples: int, env: cMDP,
                         n_depths: List[int] = [10, 20, 50, 100, 200]) -> Dict:
    """Run single experiment: train agent and measure extraction accuracy"""
    agent = ModelBasedAgent(env.n_states, env.n_actions)
    agent.train_from_random_policy(env, n_samples)

    results = {
        'n_samples': n_samples,
        'errors': [],
        'regrets': [],
        'depths': n_depths
    }

    for depth in n_depths:
        error, regret, _ = compute_metrics(agent, env, n_trials=depth)
        results['errors'].append(error)
        results['regrets'].append(regret)

    return results


def run_experiments(n_repetitions: int = 5):
    """
    Run full experimental suite matching Richens et al. Section 3.1.

    Experiments:
    1. Error vs training samples (at fixed depth)
    2. Error vs goal depth (at fixed training)
    """
    print("Creating environment...")
    env = cMDP.create_random_communicating(n_states=20, n_actions=5, max_outcomes=5)

    # Experiment 1: Vary training samples
    print("\nExperiment 1: Error vs training samples...")
    sample_sizes = [500, 1000, 2000, 3000, 5000, 7000, 10000]
    results_by_samples = {size: [] for size in sample_sizes}

    for rep in range(n_repetitions):
        print(f"  Repetition {rep + 1}/{n_repetitions}")
        for n_samples in sample_sizes:
            result = run_single_experiment(n_samples, env, n_depths=[50])
            results_by_samples[n_samples].append({
                'error': result['errors'][0],
                'regret': result['regrets'][0]
            })

    # Experiment 2: Vary goal depth
    print("\nExperiment 2: Error vs goal depth...")
    depths = [10, 20, 50, 75, 100, 200, 400]
    results_by_depth = {depth: [] for depth in depths}

    fixed_samples = 5000
    for rep in range(n_repetitions):
        print(f"  Repetition {rep + 1}/{n_repetitions}")
        result = run_single_experiment(fixed_samples, env, n_depths=depths)
        for i, depth in enumerate(depths):
            results_by_depth[depth].append({
                'error': result['errors'][i],
                'regret': result['regrets'][i]
            })

    return results_by_samples, results_by_depth, env


def plot_results(results_by_samples, results_by_depth):
    """Generate plots matching Figure 3 from Richens et al."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Error vs goal depth (matching Figure 3a)
    depths = sorted(results_by_depth.keys())
    mean_errors = []
    std_errors = []

    for depth in depths:
        errors = [r['error'] for r in results_by_depth[depth]]
        mean_errors.append(np.mean(errors))
        std_errors.append(np.std(errors) / np.sqrt(len(errors)) * 1.96)  # 95% CI

    ax1.errorbar(depths, mean_errors, yerr=std_errors, marker='o',
                 capsize=5, capthick=2, linewidth=2, markersize=8)

    # Fit O(n^{-1/2}) curve
    depths_fit = np.array(depths)
    # Filter out very small values for fitting
    valid_idx = np.array(mean_errors) > 0.01
    if valid_idx.any():
        params = np.polyfit(np.log(depths_fit[valid_idx]),
                          np.log(np.array(mean_errors)[valid_idx]), 1)
        fit_curve = np.exp(params[1]) * depths_fit ** params[0]
        ax1.plot(depths, fit_curve, '--', linewidth=2, alpha=0.7,
                label=f'Fit: y = {np.exp(params[1]):.3f} × n^{params[0]:.2f}')

    ax1.set_xlabel('Goal Depth (n)', fontsize=12)
    ax1.set_ylabel('Mean Error ⟨ε⟩', fontsize=12)
    ax1.set_title('Error vs Goal Depth (N_samples = 5000)', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Error vs regret (matching Figure 3b)
    sample_sizes = sorted(results_by_samples.keys())
    mean_errors_samples = []
    std_errors_samples = []
    mean_regrets = []

    for size in sample_sizes:
        errors = [r['error'] for r in results_by_samples[size]]
        regrets = [r['regret'] for r in results_by_samples[size]]
        mean_errors_samples.append(np.mean(errors))
        std_errors_samples.append(np.std(errors) / np.sqrt(len(errors)) * 1.96)
        mean_regrets.append(np.mean(regrets))

    ax2.errorbar(mean_regrets, mean_errors_samples, yerr=std_errors_samples,
                 marker='o', capsize=5, capthick=2, linewidth=2, markersize=8)

    # Add sample size labels
    for i, size in enumerate(sample_sizes):
        if i % 2 == 0:  # Label every other point to avoid crowding
            ax2.annotate(f'{size}',
                        xy=(mean_regrets[i], mean_errors_samples[i]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=9, alpha=0.7)

    ax2.set_xlabel('Mean Regret ⟨δ⟩ (at depth n=50)', fontsize=12)
    ax2.set_ylabel('Mean Error ⟨ε⟩', fontsize=12)
    ax2.set_title('Error vs Regret', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
