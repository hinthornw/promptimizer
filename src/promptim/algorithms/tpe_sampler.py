# Adapted from Optuna. All credit go to the authors of that library.
# https://optuna.readthedocs.io/en/stable/_modules/optuna/samplers/_tpe/sampler.html#TPESampler
import math
import random
from typing import List, Dict, Tuple, Callable, Awaitable

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
        self.observations: Dict[str, List[Tuple[float, float]]] = {}
        # You can store advanced settings here if desired (bandwidth, etc.)

    def register(self, param_name: str, value: float, objective: float):
        """
        Add one completed trial's param value and objective outcome.
        """
        if param_name not in self.observations:
            self.observations[param_name] = []
        self.observations[param_name].append((value, objective))

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

        # 1) Sort by objective
        #    If lower_is_better => sort ascending, else descending
        sorted_history = sorted(
            history, key=lambda x: x[1], reverse=(not lower_is_better)
        )

        # 2) Split into 'good' vs 'bad'
        n_good = max(1, int(math.ceil(len(sorted_history) * gamma)))
        good = sorted_history[:n_good]  # top fraction
        bad = sorted_history[n_good:]  # rest

        # 3) We'll now sample from the 'good' mixture. We generate n_candidates points.
        #    For each candidate x, we compute logpdf_good(x) and logpdf_bad(x),
        #    pick the x maximizing [logpdf_good - logpdf_bad].
        best_x = None
        best_obj = _MIN  # we want max
        for _ in range(n_candidates):
            x_cand = self._sample_from_mixture(good, low, high, bandwidth)
            log_l_good = self._log_mixture_pdf(x_cand, good, bandwidth)
            log_l_bad = self._log_mixture_pdf(x_cand, bad, bandwidth)
            # ratio => log(l(x)/g(x)) = log l(x) - log g(x)
            ratio = log_l_good - log_l_bad
            if ratio > best_obj:
                best_obj = ratio
                best_x = x_cand

        if best_x is None:
            return self.rng.uniform(low, high)

        best_x = max(low, min(high, best_x))
        return best_x

    def suggest_int(
        self,
        param_name: str,
        low: int,
        high: int,
        n_candidates: int = 24,
        gamma: float = 0.2,
        lower_is_better: bool = True,
        bandwidth: float = 0.1,
    ) -> int:
        """Return a suggested integer value for the given param within [low, high]."""
        float_val = self.suggest(
            param_name=param_name,
            low=float(low) - 0.4999,  # ensure rounding works correctly
            high=float(high) + 0.4999,
            n_candidates=n_candidates,
            gamma=gamma,
            lower_is_better=lower_is_better,
            bandwidth=bandwidth,
        )
        return int(round(float_val))

    async def optimize(self, objective_fn: Callable[..., [Awaitable[float]]], n_trials: int = 30) -> float:
        """Run optimization for n_trials, returning best objective value found.
        The objective_fn should take this sampler as argument and return a float score.
        """
        best_score = float("-inf")
        for _ in range(n_trials):
            score = await objective_fn(self)
            best_score = max(best_score, score)
        return best_score

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

        # Randomly pick one data point in 'dataset', then sample from a Normal
        # around that data point's param value with given bandwidth.
        # This is the simplest approach (each data point has weight = 1/len).
        idx = self.rng.randint(0, len(dataset) - 1)
        center = dataset[idx][0]
        x = self.rng.gauss(center, bandwidth)
        # If you prefer a more robust bandwidth, you can do e.g. Silverman's rule
        # or Freedman-Diaconis. For simplicity, we use fixed bandwidth.
        return x

    def _log_mixture_pdf(
        self, x: float, dataset: List[Tuple[float, float]], bandwidth: float
    ) -> float:
        """mixture is average of Normal(center=each data point, sigma=bandwidth)."""
        if not dataset:
            return _MIN

        # log of average => log(1/N * sum(pdf_i(x))) => log sum(pdf_i(x)) - log N
        # We'll do it carefully to avoid floating underflow by using log-sum-exp approach.
        log_vals = []
        for val, _ in dataset:
            log_vals.append(self._log_normal_pdf(x, val, bandwidth))

        # log-sum-exp
        mx = max(log_vals)
        s = 0.0
        for lv in log_vals:
            s += math.exp(lv - mx)
        return mx + math.log(s) - math.log(len(log_vals))

    def _log_normal_pdf(self, x: float, mu: float, sigma: float) -> float:
        if sigma <= 0.0:
            return _MIN
        c = math.log(1.0 / (math.sqrt(2.0 * math.pi) * sigma))
        z = (x - mu) / sigma
        return c - 0.5 * z * z
