from pomdp import SimplePOMDP, PretrainedPOMDPAgent
from typing import Tuple, List
import matplotlib.pyplot as plt



def realize_history_with_reset(env: SimplePOMDP,
                               target_history: List[Tuple[int, int]]) -> Tuple[bool, int]:
    """
    Try to realize target history starting from episodic reset.

    Returns:
        (success, final_state)
    """
    state = env.reset()  # Episodic reset!

    for target_obs, target_action in target_history:
        # Take the action specified in history
        next_state, observation = env.step(state, target_action)

        # Check if we observed what we needed
        if observation != target_obs:
            return False, state  # Failed to realize history

        state = next_state

    return True, state  # Successfully realized history!


def extract_kernel_with_episodic_resets(agent: PretrainedPOMDPAgent,
                                       env: SimplePOMDP,
                                       history: List[Tuple[int, int]],
                                       action: int,
                                       target_obs: int,
                                       n_trials: int = 50,
                                       max_attempts: int = 10000) -> Tuple[float, int, float]:
    """
    TRUE extraction: Use episodic resets to gather n valid trials.

    Returns:
        (estimated_prob, actual_attempts, coverage_estimate)
    """
    successes = 0
    attempts = 0
    valid_trials = 0

    while valid_trials < n_trials and attempts < max_attempts:
        attempts += 1

        # EPISODIC RESET
        realized, final_state = realize_history_with_reset(env, history)

        if realized:
            # We successfully reached history h!
            # Now take the action we want to test
            _, observation = env.step(final_state, action)

            # Record if we saw target observation
            if observation == target_obs:
                successes += 1

            valid_trials += 1

    if valid_trials < n_trials:
        print(f"Warning: Only got {valid_trials}/{n_trials} valid trials after {max_attempts} attempts")

    # Estimate probability
    p_hat = successes / valid_trials if valid_trials > 0 else 0.0

    # Estimate coverage
    rho_estimate = valid_trials / attempts if attempts > 0 else 0.0

    return p_hat, attempts, rho_estimate


def extract_kernel_with_switch_goals(agent: PretrainedPOMDPAgent,
                                     env: SimplePOMDP,
                                     history: List[Tuple[int, int]],
                                     action: int,
                                     target_obs: int,
                                     n_trials: int = 50,
                                     max_attempts: int = 10000) -> Tuple[float, int]:
    """
    Extract using binary switch goals (like Algorithm 2).
    This is what the paper actually does!
    """
    alt_action = (action + 1) % env.n_actions

    # For each threshold k, determine which goal agent prefers
    k_star = n_trials

    for k in range(1, n_trials + 1):
        chosen = agent.optimal_action_for_goal(
            history, action, alt_action, target_obs, n_trials, k
        )
        if chosen == action:
            k_star = k
            break

    p_hat = (k_star - 0.5) / n_trials

    # But we still need to count attempts for coverage
    # (In reality, agent's decision is based on learned model,
    # but gathering n trials still requires episodes)
    _, attempts, _ = extract_kernel_with_episodic_resets(
        agent, env, history, action, target_obs, n_trials, max_attempts
    )

    return p_hat, attempts


def run_extraction_experiment():
    """
    Run experiment showing the role of episodic resets.
    """
    print("="*70)
    print("TRUE POMDP EXTRACTION WITH EPISODIC RESETS")
    print("="*70)

    env = SimplePOMDP(n_states=5, n_actions=2, n_observations=3)
    agent = PretrainedPOMDPAgent(env)

    # Test histories of different lengths
    
    test_histories = [
        [],
        [(1, 0)],
        [(1, 0), (2, 1)],
        [(0, 0), (1, 1)],
    ]

    n_trials_list = [10, 20, 50, 100]

    results = []

    print("\nTesting extraction with different history lengths and trial counts...")
    print()

    for history in test_histories:
        print(f"History (length={len(history)}): {history}")

        # Compute true kernel
        true_kernel = env.compute_kernel(history, action=0)
        target_obs = 0  # Test for first observation
        true_p = true_kernel[target_obs]

        print(f"  True K(h,0) = {true_kernel}")
        print(f"  True P(o={target_obs}|h,a=0) = {true_p:.3f}")

        for n_trials in n_trials_list:
            # Extract using episodic resets
            p_hat, attempts, rho = extract_kernel_with_episodic_resets(
                agent, env, history, action=0, target_obs=target_obs,
                n_trials=n_trials, max_attempts=5000
            )

            error = abs(p_hat - true_p)

            print(f"    n={n_trials:3d}: p̂={p_hat:.3f}, error={error:.3f}, "
                  f"attempts={attempts:4d}, ρ(h)≈{rho:.3f}")

            results.append({
                'history_length': len(history),
                'n_trials': n_trials,
                'true_p': true_p,
                'estimated_p': p_hat,
                'error': error,
                'attempts': attempts,
                'coverage': rho
            })

        print()

    return results


def plot_extraction_results(results):
    """Plot results showing sample complexity"""
    import pandas as pd
    df = pd.DataFrame(results)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Error vs n_trials for different history lengths
    ax1 = axes[0, 0]
    for length in df['history_length'].unique():
        data = df[df['history_length'] == length]
        ax1.plot(data['n_trials'], data['error'], marker='o',
                label=f'Length {length}', linewidth=2, markersize=8)

    ax1.set_xlabel('Number of Valid Trials (n)', fontsize=11)
    ax1.set_ylabel('Extraction Error', fontsize=11)
    ax1.set_title('Error vs Valid Trials', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')

    # Plot 2: Attempts needed vs n_trials for different lengths
    ax2 = axes[0, 1]
    for length in df['history_length'].unique():
        data = df[df['history_length'] == length]
        ax2.plot(data['n_trials'], data['attempts'], marker='o',
                label=f'Length {length}', linewidth=2, markersize=8)

    ax2.set_xlabel('Valid Trials Needed (n)', fontsize=11)
    ax2.set_ylabel('Episodes Attempted', fontsize=11)
    ax2.set_title('Sample Complexity: Episodes vs Trials', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    ax2.set_yscale('log')

    # Plot 3: Coverage by history length
    ax3 = axes[1, 0]
    coverage_by_length = df.groupby('history_length')['coverage'].mean()
    ax3.bar(coverage_by_length.index, coverage_by_length.values,
           color='orange', alpha=0.7, width=0.6)
    ax3.set_xlabel('History Length', fontsize=11)
    ax3.set_ylabel('Coverage ρ(h)', fontsize=11)
    ax3.set_title('Coverage Decay with History Length', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    # Plot 4: Sample complexity ratio (attempts / n_trials)
    ax4 = axes[1, 1]
    df['complexity_ratio'] = df['attempts'] / df['n_trials']
    for length in df['history_length'].unique():
        data = df[df['history_length'] == length]
        ax4.plot(data['n_trials'], data['complexity_ratio'], marker='o',
                label=f'Length {length}', linewidth=2, markersize=8)

    ax4.set_xlabel('Valid Trials (n)', fontsize=11)
    ax4.set_ylabel('Episodes / Trials = 1/ρ(h)', fontsize=11)
    ax4.set_title('Sample Complexity Ratio', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Ideal (no coverage loss)')

    plt.tight_layout()
    plt.savefig('/workspace/projects/pomdp/code/main/results/pomdp_extraction_results.png', dpi=300, bbox_inches='tight')
    return fig

