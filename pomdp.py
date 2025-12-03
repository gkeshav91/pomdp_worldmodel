import numpy as np
from typing import Tuple, List
from scipy.stats import binom


class SimplePOMDP:
    """Simple POMDP for testing extraction"""
    def __init__(self, n_states=5, n_actions=2, n_observations=3):
        self.n_states = n_states
        self.n_actions = n_actions
        self.n_observations = n_observations

        # Create simple structured POMDP
        self.transitions = np.zeros((n_actions, n_states, n_states))
        self.emissions = np.zeros((n_actions, n_states, n_observations))

        # Simple ring transitions
        for a in range(n_actions):
            for s in range(n_states):
                # Action 0: move forward
                # Action 1: move backward or stay
                if a == 0:
                    next_s = (s + 1) % n_states
                    self.transitions[a, s, next_s] = 0.8
                    self.transitions[a, s, s] = 0.2
                else:
                    next_s = (s - 1) % n_states
                    self.transitions[a, s, next_s] = 0.7
                    self.transitions[a, s, s] = 0.3

        # Observations: some aliasing
        for a in range(n_actions):
            for s in range(n_states):
                primary_obs = s % n_observations
                self.emissions[a, s, primary_obs] = 0.7
                # Uniform noise on others
                for o in range(n_observations):
                    if o != primary_obs:
                        self.emissions[a, s, o] = 0.3 / (n_observations - 1)

        self.initial_belief = np.ones(n_states) / n_states

    def reset(self) -> int:
        """Reset to initial state"""
        return np.random.choice(self.n_states, p=self.initial_belief)

    def step(self, state: int, action: int) -> Tuple[int, int]:
        """Take action, return (next_state, observation)"""
        next_state = np.random.choice(self.n_states, p=self.transitions[action, state, :])
        observation = np.random.choice(self.n_observations, p=self.emissions[action, next_state, :])
        return next_state, observation

    def compute_kernel(self, history: List[Tuple[int, int]], action: int) -> np.ndarray:
        """Compute true K(h,a)(o) = P(o|h,a)"""
        # Compute belief after history
        belief = self.initial_belief.copy()
        for obs, act in history:
            # Bayesian update
            b_pred = self.transitions[act, :, :].T @ belief
            b_new = self.emissions[act, :, obs] * b_pred
            if b_new.sum() > 0:
                belief = b_new / b_new.sum()

        # Compute predictive kernel
        kernel = np.zeros(self.n_observations)
        for s in range(self.n_states):
            for s_p in range(self.n_states):
                for o in range(self.n_observations):
                    kernel[o] += (belief[s] *
                                 self.transitions[action, s, s_p] *
                                 self.emissions[action, s_p, o])

        return kernel


class PretrainedPOMDPAgent:
    """
    Agent that has learned a world model (details don't matter).
    For testing extraction, we'll give it the true model.
    """
    def __init__(self, env: SimplePOMDP):
        self.env = env
        # Pretend agent has learned the model perfectly
        # (In reality, would use EM/particle filters/etc)
        self.learned_kernel_cache = {}

    def get_kernel_estimate(self, history: List[Tuple[int, int]], action: int) -> np.ndarray:
        """Agent's estimate of K(h,a)(o)"""
        key = (tuple(history), action)
        if key not in self.learned_kernel_cache:
            # Agent uses its learned model to compute kernel
            # For testing, we use true model with some noise
            true_kernel = self.env.compute_kernel(history, action)
            noise = np.random.normal(0, 0.05, len(true_kernel))
            noisy_kernel = true_kernel + noise
            noisy_kernel = np.clip(noisy_kernel, 0, 1)
            noisy_kernel = noisy_kernel / noisy_kernel.sum()
            self.learned_kernel_cache[key] = noisy_kernel
        return self.learned_kernel_cache[key]

    def optimal_action_for_goal(self, history: List[Tuple[int, int]],
                               action_a: int, action_b: int,
                               target_obs: int, n_trials: int,
                               threshold: int) -> int:
        """Which action achieves goal better?"""
        kernel = self.get_kernel_estimate(history, action_a)
        p = kernel[target_obs]

        prob_a = binom.cdf(threshold, n_trials, p)
        prob_b = 1 - prob_a

        return action_a if prob_a >= prob_b else action_b

