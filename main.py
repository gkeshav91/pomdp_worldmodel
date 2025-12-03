import numpy as np
import matplotlib.pyplot as plt
from helper_mdp import run_experiments, plot_results
from helper_pomdp import run_extraction_experiment, plot_extraction_results

np.random.seed(42)
run_mdp = False
run_pomdp = True

#------------------------------------run mdp experiments------------------------------------------------------#
if run_mdp:
    print("=" * 70)
    print("Richens et al. Experiment: World Models in General Agents")
    print("=" * 70)

    # Run experiments (use fewer repetitions for faster testing)
    results_samples, results_depth, env = run_experiments(n_repetitions=3)

    print("\n" + "=" * 70)
    print("Generating plots...")
    fig = plot_results(results_samples, results_depth)
    plt.savefig('/workspace/projects/pomdp/code/main/results/richens_experiments_results.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'richens_experiments_results.png'")

    # Print summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    print("\nError vs Training Samples (at depth=50):")
    for size in sorted(results_samples.keys()):
        errors = [r['error'] for r in results_samples[size]]
        print(f"  N={size:5d}: Error = {np.mean(errors):.4f} ± {np.std(errors):.4f}")

    print("\nError vs Goal Depth (N_samples=5000):")
    for depth in sorted(results_depth.keys()):
        errors = [r['error'] for r in results_depth[depth]]
        print(f"  Depth={depth:3d}: Error = {np.mean(errors):.4f} ± {np.std(errors):.4f}")

    print("\n" + "=" * 70)
    print("Experiments complete!")
    print("=" * 70)

    plt.show()

#------------------------------------run pomdp experiments------------------------------------------------------#



if run_pomdp:
    results = run_extraction_experiment()

    print("\n" + "="*70)
    print("SUMMARY: Sample Complexity Analysis")
    print("="*70)

    import pandas as pd
    df = pd.DataFrame(results)

    print("\nSample complexity (attempts / valid_trials) by history length:")
    for length in sorted(df['history_length'].unique()):
        data = df[df['history_length'] == length]
        mean_ratio = (data['attempts'] / data['n_trials']).mean()
        mean_coverage = data['coverage'].mean()
        print(f"  Length {length}: {mean_ratio:.1f}× episodes needed "
              f"(ρ(h) ≈ {mean_coverage:.3f})")

    print("\n" + "="*70)
    print("KEY INSIGHTS:")
    print("  • Each 'valid trial' requires 1/ρ(h) episodes on average")
    print("  • Longer histories → lower ρ(h) → more episodes needed")
    print("  • This is the POMDP-specific sample complexity!")
    print("  • Without episodic resets, extraction is impossible")
    print("="*70)

    fig = plot_extraction_results(results)
    plt.savefig('true_pomdp_extraction.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved as 'true_pomdp_extraction.png'")
    plt.show()