import numpy as np
from typing import List, Tuple, Dict, Any
from pomdp import SyntheticPOMDP, PretrainedPOMDPAgent


def generate_test_histories(pomdp_instance, samples_per_length=15, max_len=4):
    """
    Generates (1 empty history + samples_per_length * max_len) histories: 
    """
    all_histories = [[]] # Start with the empty history
    for length in range(1, max_len + 1):
        for _ in range(samples_per_length):
            history = []
            current_state = np.random.choice( pomdp_instance.S,  p=pomdp_instance.initial_belief )                        
            for _ in range(length):
                action = np.random.choice(pomdp_instance.A)                
                obs, next_state = pomdp_instance.step(current_state, action)
                history.append((obs, action))
                current_state = next_state            
            all_histories.append(history)
            
    return all_histories

def generate_test_histories_maxlen(pomdp_instance, samples_per_length=15, max_len=4):
    """
    Generates (1 empty history + samples_per_length * max_len) histories: 
    """
    all_histories = [] # Start with the empty history
    for _ in range(samples_per_length):
        history = []
        current_state = np.random.choice( pomdp_instance.S,  p=pomdp_instance.initial_belief )                        
        for _ in range(max_len):
            action = np.random.choice(pomdp_instance.A)                
            obs, next_state = pomdp_instance.step(current_state, action)
            history.append((obs, action))
            current_state = next_state            
        all_histories.append(history)
            
    return all_histories


def learn_kernel_empirically(
    env: SyntheticPOMDP,
    history: List[Tuple[int, int]],
    action: int,
    n_valid_trials: int = 100,
    max_attempts: int = 1000000,
    dirichlet_alpha: float = 1.0
) -> Dict[str, Any]:
    """
    Learns the transition kernel P(o|h,a) by brute-force simulation.
    Returns the learned kernel and the difficulty (realization rate) of the history.
    """
    
    def realize_history_once():
        state = np.random.choice(env.S, p=env.initial_belief)
        for target_o, target_a in history:
            obs, next_state = env.step(state, target_a)
            if obs != target_o:
                return False, None
            state = next_state            
        return True, state

    observation_counts = np.zeros(env.O, dtype=float)
    attempts = 0
    valid = 0

    while valid < n_valid_trials and attempts < max_attempts:
        attempts += 1
        realized, end_state = realize_history_once()

        if realized:
            obs, _ = env.step(end_state, action)
            observation_counts[obs] += 1.0
            valid += 1

    smoothed_counts = observation_counts + dirichlet_alpha
    learned_kernel = smoothed_counts / smoothed_counts.sum()    
    realization_rate = valid / attempts if attempts > 0 else 0.0

    return { "learned_kernel": learned_kernel, "realization_rate": realization_rate, "attempts": attempts }


def calculate_tv_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Computes the Total Variation Distance: 0.5 * sum(|p - q|)"""
    return 0.5 * np.sum(np.abs(p - q))



def extract_kernel_with_switch_goals(
    agent: PretrainedPOMDPAgent,
    env: SyntheticPOMDP,
    history: List[Tuple[int, int]],
    action: int,
    target_obs: int,
    n_trials: int = 50
) -> float:
    """
    Extracts the agent's internal P(o|h,a) by finding the k_star flip point.
    """
    alt_action = (action + 1) % env.A
    k_star = n_trials

    # Sweep through thresholds to find where the agent flips from alt_action to action
    for k in range(1, n_trials + 1):
        chosen = agent.optimal_action_for_goal( history, action, alt_action, target_obs, n_trials, k )
        if chosen == action:
            k_star = k
            break

    # Continuity correction for p estimate
    p_hat = (k_star - 0.5) / n_trials
    return p_hat


def learn_kernel_empirically_approximation(
    env: SyntheticPOMDP,
    history: List[Tuple[int, int]],
    action: int,
    n_valid_trials: int = 100,
    max_attempts: int = 1000000,
    dirichlet_alpha: float = 1.0,
    approximation_factor: float = 0.005,
    max_history_length: int = 5
) -> Dict[str, Any]:
    """
    Learns the transition kernel P(o|h,a) by brute-force simulation.
    Returns the learned kernel and the difficulty (realization rate) of the history.
    """
    target_belief = env.compute_belief(history)

    current_state = np.random.choice(env.S, p=env.initial_belief)

    observation_counts = np.zeros(env.O, dtype=float)
    attempts = 0
    valid = 0
    cur_his = []

    while valid < n_valid_trials and attempts < max_attempts:
        
        action_random = np.random.choice(env.A)
        obs, next_state = env.step(current_state, action_random)
        cur_his.append((obs, action_random))
        cur_belief = env.compute_belief(cur_his)
        attempts += 1

        diff_belief = calculate_tv_distance(target_belief, cur_belief)
        if diff_belief < approximation_factor:
            obs, _ = env.step(next_state, action)
            observation_counts[obs] += 1.0
            valid += 1
            cur_his = []
            current_state = np.random.choice(env.S, p=env.initial_belief)
        else:
            current_state = next_state

        if len(cur_his) >= max_history_length:
            cur_his = []
            current_state = np.random.choice(env.S, p=env.initial_belief)

    smoothed_counts = observation_counts + dirichlet_alpha
    learned_kernel = smoothed_counts / smoothed_counts.sum()    
    realization_rate = valid / attempts if attempts > 0 else 0.0

    return { "learned_kernel": learned_kernel, "realization_rate": realization_rate, "attempts": attempts }

