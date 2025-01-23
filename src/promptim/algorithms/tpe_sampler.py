# Adapted from Optuna. All credit go to the authors of that library.
# https://optuna.readthedocs.io/en/stable/_modules/optuna/samplers/_tpe/sampler.html#TPESampler
import math
import random
from typing import List, Dict, Tuple, Callable, Awaitable, Union, Any

_MIN = -999999999.0


class TPESampler:
    """Tree-structured parzen estimator; based on Optuna's implementation but without the extra power.

    For each parameter, we store (value, objective) for each completed trial.
    We then:
      1) Sort by objective (assume 'lower is better' by default).
      2) Split into 'good' set (best fraction) vs 'bad' set (rest).
      3) Model each set as a mixture of Gaussians (one Gaussian per data point).
      4) Generate multiple candidate points from the mixture of 'good' set,
         evaluating ratio l(x)/g(x), and choose the x that maximizes it.
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        # Data structure to store param -> list of (value, objective)
        self.observations: Dict[str, List[Tuple[Union[float, int, str], float]]] = {}
        # You can store advanced settings here if desired (bandwidth, etc.)

    def register(
        self, param_name: str, value: Union[float, int, str], objective: float
    ):
        """
        Add one completed trial's param value and objective outcome.
        """
        if param_name not in self.observations:
            self.observations[param_name] = []
        self.observations[param_name].append((value, objective))

    def suggest_categorical(
        self,
        param_name: str,
        choices: List[Any],
        n_candidates: int = 24,
        gamma: float = 0.2,
        lower_is_better: bool = True,
    ) -> Any:
        """Return a suggested categorical value for the given param."""
        history = self.observations.get(param_name, [])
        if len(history) < 2:
            return self.rng.choice(choices)

        sorted_history = sorted(
            history, key=lambda x: x[1], reverse=(not lower_is_better)
        )

        n_good = max(1, int(math.ceil(len(sorted_history) * gamma)))
        good = sorted_history[:n_good]
        bad = sorted_history[n_good:]

        good_counts = {choice: 0.0 for choice in choices}
        bad_counts = {choice: 0.0 for choice in choices}

        pseudocount = 1.0
        for choice in choices:
            good_counts[choice] = pseudocount
            bad_counts[choice] = pseudocount

        for val, _ in good:
            good_counts[val] += 1.0
        for val, _ in bad:
            bad_counts[val] += 1.0

        good_total = sum(good_counts.values())
        bad_total = sum(bad_counts.values())

        for choice in choices:
            good_counts[choice] /= good_total
            bad_counts[choice] /= bad_total

        best_choice = None
        best_ratio = _MIN

        for _ in range(n_candidates):
            candidate = self.rng.choice(choices)
            ratio = math.log(good_counts[candidate]) - math.log(bad_counts[candidate])

            if ratio > best_ratio:
                best_ratio = ratio
                best_choice = candidate

        return best_choice if best_choice is not None else self.rng.choice(choices)

    def suggest(
        self,
        param_name: str,
        low: float,
        high: float,
        n_candidates: int = 24,
        gamma: float = 0.2,
        lower_is_better: bool = True,
        bandwidth: float = 0.1,
    ) -> float:
        """Return a suggested float value for the given param within [low, high].

        Args:
            n_candidates: Number of candidate samples from the 'good' mixture
            gamma: Fraction of trials to consider 'good' (0.2 => top 20%).
            lower_is_better: If True, smaller objective is better. If False, bigger is better.
            bandwidth: Kernel width (std dev) for each sample-based Gaussian in the mixture.
        """
        history = self.observations.get(param_name, [])
        if len(history) < 2:
            return self.rng.uniform(low, high)

        sorted_history = sorted(
            history, key=lambda x: x[1], reverse=(not lower_is_better)
        )

        n_good = max(1, int(math.ceil(len(sorted_history) * gamma)))
        good = sorted_history[:n_good]
        bad = sorted_history[n_good:]

        best_x = None
        best_obj = _MIN

        for _ in range(n_candidates):
            x_cand = self._sample_from_mixture(good, low, high, bandwidth)
            log_l_good = self._log_mixture_pdf(x_cand, good, bandwidth)
            log_l_bad = self._log_mixture_pdf(x_cand, bad, bandwidth)
            ratio = log_l_good - log_l_bad

            if ratio > best_obj:
                best_obj = ratio
                best_x = x_cand

        if best_x is None:
            return self.rng.uniform(low, high)

        return max(low, min(high, best_x))

    def suggest_int(
        self,
        param_name: str,
        low: int,
        high: int,
        n_candidates: int = 24,
        gamma: float = 0.2,
        lower_is_better: bool = True,
    ) -> int:
        """Return a suggested integer value for the given param within [low, high]."""
        float_val = self.suggest(
            param_name=param_name,
            low=float(low) - 0.4999,
            high=float(high) + 0.4999,
            n_candidates=n_candidates,
            gamma=gamma,
            lower_is_better=lower_is_better,
        )
        return int(round(float_val))

    async def optimize(
        self, objective_fn: Callable[[Any], Awaitable[float]], n_trials: int = 30
    ) -> "Trial":
        """Run optimization for n_trials, returning best trial."""
        best_score = float("-inf")
        best_trial = None

        for _ in range(n_trials):
            trial = Trial(self)
            score = await objective_fn(trial)

            if score > best_score:
                best_score = score
                best_trial = trial

        return best_trial

    def _sample_from_mixture(
        self,
        dataset: List[Tuple[float, float]],
        low: float,
        high: float,
        bandwidth: float,
    ) -> float:
        """
        Sample one x from the mixture of Gaussians, each centered on a
        data point from `dataset`.
        """
        if not dataset:
            return self.rng.uniform(low, high)

        idx = self.rng.randint(0, len(dataset) - 1)
        center = dataset[idx][0]

        min_distance = min(center - low, high - center)
        adj_bandwidth = min(bandwidth, min_distance / 3)

        return self.rng.gauss(center, adj_bandwidth)

    def _log_mixture_pdf(
        self, x: float, dataset: List[Tuple[float, float]], bandwidth: float
    ) -> float:
        """mixture is average of Normal(center=each data point, sigma=bandwidth)."""
        if not dataset:
            return _MIN

        log_vals = []
        for val, _ in dataset:
            log_vals.append(self._log_normal_pdf(x, val, bandwidth))

        max_log = max(log_vals)
        sum_exp = 0.0
        for log_val in log_vals:
            sum_exp += math.exp(log_val - max_log)

        return max_log + math.log(sum_exp) - math.log(len(log_vals))

    def _log_normal_pdf(self, x: float, mu: float, sigma: float) -> float:
        if sigma <= 0.0:
            return _MIN

        z = (x - mu) / sigma
        return -0.5 * z * z - math.log(sigma) - 0.5 * math.log(2 * math.pi)


class Trial:
    def __init__(self, sampler: TPESampler):
        self.sampler = sampler
        self.params = {}

    def suggest_categorical(
        self,
        name: str,
        choices: List[Any],
        n_candidates: int = 24,
        gamma: float = 0.2,
        lower_is_better: bool = True,
    ) -> Any:
        value = self.sampler.suggest_categorical(
            name,
            choices,
            n_candidates=n_candidates,
            gamma=gamma,
            lower_is_better=lower_is_better,
        )
        self.params[name] = value
        return value

    def suggest_int(
        self,
        name: str,
        low: int,
        high: int,
        n_candidates: int = 24,
        gamma: float = 0.2,
        lower_is_better: bool = True,
    ) -> int:
        value = self.sampler.suggest_int(
            name,
            low,
            high,
            n_candidates=n_candidates,
            gamma=gamma,
            lower_is_better=lower_is_better,
        )
        self.params[name] = value
        return value
