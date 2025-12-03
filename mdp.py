import numpy as np
from dataclasses import dataclass
from scipy.stats import binom

@dataclass
class cMDP:
    """Controlled Markov Decision Process (MDP without rewards)"""
    n_states: int
    n_actions: int
    transitions: np.ndarray  # Shape: (n_actions, n_states, n_states)

    def __post_init__(self):
        # Verify transitions are valid probability distributions
        assert self.transitions.shape == (self.n_actions, self.n_states, self.n_states)
        assert np.allclose(self.transitions.sum(axis=2), 1.0)

    @staticmethod
    def create_random_communicating(n_states: int = 20, n_actions: int = 5,
                                   max_outcomes: int = 5) -> 'cMDP':
        """
        Create a random communicating cMDP with sparse transitions.
        Each state-action pair has at most max_outcomes non-zero transitions.
        """
        transitions = np.zeros((n_actions, n_states, n_states))

        for a in range(n_actions):
            for s in range(n_states):
                # Randomly select which outcomes are possible
                n_outcomes = min(max_outcomes, n_states)
                possible_outcomes = np.random.choice(n_states, size=n_outcomes, replace=False)

                # Generate random probabilities
                probs = np.random.dirichlet(np.ones(n_outcomes))

                # Assign probabilities
                for outcome, prob in zip(possible_outcomes, probs):
                    transitions[a, s, outcome] = prob

        # Ensure communicating by adding small probability of reaching any state
        # from any other state (relaxing strict sparsity slightly)
        epsilon = 0.01
        for a in range(n_actions):
            for s in range(n_states):
                # Add small uniform probability to all states
                transitions[a, s, :] = (1 - epsilon) * transitions[a, s, :] + epsilon / n_states

        return cMDP(n_states, n_actions, transitions)

    def sample_transition(self, state: int, action: int) -> int:
        """Sample next state given current state and action"""
        return np.random.choice(self.n_states, p=self.transitions[action, state, :])

    def get_transition_prob(self, state: int, action: int, next_state: int) -> float:
        """Get P(s'|s,a)"""
        return self.transitions[action, state, next_state]


class ModelBasedAgent:
    """
    Model-based agent that learns transition probabilities from experience.
    Uses maximum likelihood estimation with Laplace smoothing.
    """
    def __init__(self, n_states: int, n_actions: int, smoothing: float = 0.1):
        self.n_states = n_states
        self.n_actions = n_actions
        self.smoothing = smoothing

        # Count transitions: counts[a, s, s']
        self.counts = np.ones((n_actions, n_states, n_states)) * smoothing

        # Learned transition model
        self.transitions = None
        self._update_model()

    def observe_transition(self, state: int, action: int, next_state: int):
        """Update counts with observed transition"""
        self.counts[action, state, next_state] += 1

    def _update_model(self):
        """Update transition probabilities from counts"""
        self.transitions = self.counts / self.counts.sum(axis=2, keepdims=True)

    def train_from_random_policy(self, env: cMDP, n_samples: int):
        """Train agent by sampling random trajectories"""
        state = np.random.randint(env.n_states)

        for _ in range(n_samples):
            action = np.random.randint(env.n_actions)
            next_state = env.sample_transition(state, action)

            self.observe_transition(state, action, next_state)
            state = next_state

        self._update_model()

    def get_transition_prob(self, state: int, action: int, next_state: int) -> float:
        """Get learned P(s'|s,a)"""
        return self.transitions[action, state, next_state]

    def optimal_action_for_goal(self, state: int, goal_a: int, goal_b: int,
                               target_state: int, success_state: int,
                               n_trials: int, threshold: int) -> int:
        """
        Determine optimal action for composite goal using BINOMIAL CDF.

        Goal A (ψ_a): Succeed at most 'threshold' times out of n_trials
        Goal B (ψ_b): Succeed more than 'threshold' times out of n_trials

        Returns goal_a or goal_b based on which is more achievable.
        """
        # Get probability of success: P(success_state | target_state, action)
        p_success = self.get_transition_prob(target_state, goal_a, success_state)

        # Compute probabilities using binomial CDF
        # P(X ≤ threshold) where X ~ Binomial(n_trials, p_success)
        prob_goal_a = binom.cdf(threshold, n_trials, p_success)

        # P(X > threshold) = 1 - P(X ≤ threshold)
        prob_goal_b = 1 - prob_goal_a

        # Agent chooses goal with higher success probability
        if prob_goal_a >= prob_goal_b:
            return goal_a
        else:
            return goal_b

    def evaluate_composite_goal_probability(self, initial_state: int,
                                           target_state: int, action: int,
                                           success_state: int, n_trials: int,
                                           threshold: int, mode: str = 'at_most') -> float:
        """
        Evaluate the probability that composite goal is satisfied.

        This models the full composite goal structure:
        1. Start at initial_state
        2. Eventually reach target_state (using deterministic policy)
        3. Take action, observe if transition to success_state
        4. Repeat n_trials times
        5. Count successes and check if ≤ threshold (at_most) or > threshold (more_than)

        Returns probability that goal is satisfied.
        """
        # Probability of success on each trial
        p_success = self.get_transition_prob(target_state, action, success_state)

        # Probability of returning to target_state (assumed high for communicating MDP)
        # In the paper, they construct a deterministic policy that returns with prob 1
        # We approximate this as 1.0 for simplicity
        p_return = 1.0

        # Probability that all n trials can be executed
        # (in practice, we assume we can always return to target_state)
        p_execute_all_trials = p_return ** n_trials

        # Given we execute all trials, probability of satisfying threshold
        if mode == 'at_most':
            p_satisfy_threshold = binom.cdf(threshold, n_trials, p_success)
        else:  # mode == 'more_than'
            p_satisfy_threshold = 1 - binom.cdf(threshold, n_trials, p_success)

        return p_execute_all_trials * p_satisfy_threshold

