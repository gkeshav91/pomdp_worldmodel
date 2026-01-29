import pandas as pd # Optional: for a clean summary table
from pomdp import SyntheticPOMDP, PretrainedPOMDPAgent
from helper import generate_test_histories, learn_kernel_empirically, calculate_tv_distance, extract_kernel_with_switch_goals, learn_kernel_empirically_approximation, generate_test_histories_maxlen
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def model_sqrt_n(n, c, a): return c / np.sqrt(np.maximum(n - a, 1e-9))


configs =   [ 
                { "name": "POMDP-10", "states": 10, "obs": 6, "actions": 3, "target_ent": (0.9, 0.95), "p_move": 0.7, "p_random": 0.5 },
                { "name": "POMDP-20", "states": 20, "obs": 13, "actions": 3, "target_ent": (0.9, 0.95), "p_move": 0.7, "p_random": 0.5 } 
            ]

print(f"Running experiments for {len(configs)} POMDPs")

seed = 42
np.random.seed(seed)
samples_per_length = 15
max_len = 4
n_valid_trials = 100
target_action = 0
target_obs = 0
n_samples = [10, 20, 30, 50, 70, 85, 100, 125, 150, 175, 200]
pomdps = []
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

approximation_factor = 0;

for config in configs:
    pomdp = SyntheticPOMDP( n_states=config["states"],  n_obs=config["obs"],  n_actions=config["actions"], seed=seed )
    pomdp.generate_irreducible_transitions( p_move=config["p_move"],  p_random=config["p_random"] )
    pairs, ent = pomdp.optimize_emissions( ent_range=config["target_ent"],  max_iter=100 )
    max_poss = (config["states"] * (config["states"] - 1) // 2) * config["actions"]
    pretrained_agent = PretrainedPOMDPAgent(pomdp)

    if approximation_factor > 0:
        test_histories = generate_test_histories_maxlen(pomdp, samples_per_length=samples_per_length, max_len=max_len)
    else:
        test_histories = generate_test_histories(pomdp, samples_per_length=samples_per_length, max_len=max_len)

    history_metrics = []
    for history in test_histories:
        if approximation_factor == 0:
            metrics = learn_kernel_empirically(pomdp, history, action=target_action, n_valid_trials=n_valid_trials)
        else:
            metrics = learn_kernel_empirically_approximation(pomdp, history, action=target_action, n_valid_trials=n_valid_trials, approximation_factor=approximation_factor)
        true_k = pomdp.compute_true_kernel(history, action=target_action)
        metrics["true_k"] = true_k
        metrics["history"] = history
        pretrained_agent.set_knowledge(history, target_action, metrics["learned_kernel"])

        p_hats = []
        for samples in n_samples:
            p_hat = extract_kernel_with_switch_goals(pretrained_agent, pomdp, history, target_action, target_obs, n_trials=samples)
            p_hats.append(p_hat)
        metrics["p_hats"] = p_hats
        history_metrics.append(metrics)

    pomdps.append({ "name": config["name"], "pairs": pairs, "max_poss": max_poss, "entropy": round(ent, 4), "instance": pomdp, "history_metrics": history_metrics, "agent": pretrained_agent })

for i, pomdp in enumerate(pomdps[:2]):
    ax = axes[i]
    filename = f"fit_{pomdp['name']}.pdf"
    print(f"POMDP: {pomdp['name']:<10} | Confusing Pairs: {pomdp['pairs']:<6} | Max Possibilities: {pomdp['max_poss']:<8} | Entropy: {pomdp['entropy']:<8.4f}")

    history_metrics = pomdp['history_metrics']
    metrics_by_length = defaultdict(lambda: {'attempts': [], 'rates': []})    
    p_hats_by_samples = defaultdict(lambda: {'p_hats_extract_error': [], 'total_error': []})
    for history in history_metrics:
        h_len = len(history['history'])
        true_k = history['true_k']
        learned_k = history['learned_kernel']
        metrics_by_length[h_len]['attempts'].append(history["attempts"])
        metrics_by_length[h_len]['rates'].append(history["realization_rate"])
        for iter in range(len(n_samples)):
            extracted_error = abs(history['p_hats'][iter] - learned_k[target_obs])
            total_error = abs(history['p_hats'][iter] - true_k[target_obs])
            p_hats_by_samples[iter]['p_hats_extract_error'].append( extracted_error )
            p_hats_by_samples[iter]['total_error'].append( total_error )
        
    for h_len in sorted(metrics_by_length.keys()):
        m = metrics_by_length[h_len]
        avg_att  = np.mean(m['attempts']); avg_rate = np.mean(m['rates'])
        print(f"Length: {h_len:<5} | Attempts: {avg_att:<14.1f} | Rate: {avg_rate:<16.6f}")

    total_errors = []
    for iter in range(len(n_samples)):
        mean_error = np.mean(p_hats_by_samples[iter]['p_hats_extract_error'])
        std_error = np.std(p_hats_by_samples[iter]['p_hats_extract_error'])
        print(f"Samples: {n_samples[iter]:<5} | Extract Error Mean: {mean_error:<12.4f} | Extract Error Std: {std_error:<8.4f}")
        total_error_mean = np.mean(p_hats_by_samples[iter]['total_error'])
        total_error_std = np.std(p_hats_by_samples[iter]['total_error'])
        print(f"Samples: {n_samples[iter]:<5} | Total Error Mean: {total_error_mean:<12.4f} | Total Error Std: {total_error_std:<8.4f}")
        total_errors.append(total_error_mean)


    popt1, _ = curve_fit(model_sqrt_n, n_samples, total_errors, p0=[1, 0])
    c_fit, a_fit = popt1
    sign = "+" if a_fit < 0 else "-"
    abs_a = abs(a_fit)
    fit_label = (f"$y = \\frac{{{c_fit:.4f}}}{{\\sqrt{{n {sign} {abs_a:.2f}}}}}$\n")
    # Plotting
    n_plot = np.linspace(min(n_samples), max(n_samples), 500)
    ax.scatter(n_samples, total_errors, color='black', label='Data Points')
    ax.plot(n_plot, model_sqrt_n(n_plot, *popt1), 'b-', label='Fit: Fitted Line')    

    ax.text(0.95, 0.95, fit_label, transform=ax.transAxes,
            fontsize=12, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.5, edgecolor='gray'))

    ax.set_xlabel('Number of Samples', fontsize=12)
    ax.set_ylabel('Total Error', fontsize=12)
    
    # Set subplot title
    ax.set_title(f"Fit for {pomdp['name']}")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

    learning_error_mean = np.mean([ abs(history['true_k'][target_obs] - history['learned_kernel'][target_obs]) for history in history_metrics])
    learning_error_std = np.std([ abs(history['true_k'][target_obs] - history['learned_kernel'][target_obs]) for history in history_metrics])
    print(f"Learning error mean: {learning_error_mean:<12.4f} | Learning error std: {learning_error_std:<8.4f}")



plt.tight_layout()
plt.savefig('combined_pomdp_fits.pdf', bbox_inches='tight')
plt.close()

