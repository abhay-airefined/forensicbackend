from __future__ import annotations

import hashlib
import math
from typing import Callable

import numpy as np
from scipy.stats import beta, binomtest, gaussian_kde


def stable_seed(*parts: str) -> int:
    h = hashlib.sha256("::".join(parts).encode("utf-8")).hexdigest()
    return int(h[:12], 16) % (2**32 - 1)


def clip_lr(value: float, minimum: float, maximum: float) -> float:
    if not np.isfinite(value):
        return maximum
    return float(np.clip(value, minimum, maximum))


def safe_log_lr(lr: float) -> float:
    return float(math.log(max(lr, 1e-12)))


def permutation_pvalue(observed: float, null_samples: np.ndarray, greater: bool = True) -> float:
    if len(null_samples) == 0:
        return 1.0
    if greater:
        count = np.sum(null_samples >= observed)
    else:
        count = np.sum(null_samples <= observed)
    return float((count + 1) / (len(null_samples) + 1))


def bootstrap_ci(samples: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    if len(samples) == 0:
        return (0.0, 0.0)
    lo = np.percentile(samples, 100 * (alpha / 2))
    hi = np.percentile(samples, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


def beta_binomial_lr(successes: int, trials: int, p0: float = 0.01) -> float:
    if trials == 0:
        return 1.0
    prior_alt = beta(2, 8)
    p_alt = prior_alt.mean()
    l1 = max(1e-12, (p_alt**successes) * ((1 - p_alt) ** (trials - successes)))
    l0 = max(1e-12, (p0**successes) * ((1 - p0) ** (trials - successes)))
    return l1 / l0


def density_ratio(observed: float, trained_samples: np.ndarray, null_samples: np.ndarray) -> float:
    if len(trained_samples) < 2 or len(null_samples) < 2:
        return 1.0
    # jitter to prevent variance collapse
    trained = trained_samples + np.random.normal(0, 1e-6, size=len(trained_samples))
    null = null_samples + np.random.normal(0, 1e-6, size=len(null_samples))
    d1 = gaussian_kde(trained)
    d0 = gaussian_kde(null)
    p1 = float(max(1e-12, d1.evaluate([observed])[0]))
    p0 = float(max(1e-12, d0.evaluate([observed])[0]))
    return p1 / p0


def strength_from_log_lr(log_lr: float) -> str:
    a = abs(log_lr)
    if a < 0.2:
        return "No Evidence"
    if a < 0.7:
        return "Weak"
    if a < 1.4:
        return "Moderate"
    if a < 2.3:
        return "Strong"
    if a < 3.5:
        return "Very Strong"
    return "Decisive"
