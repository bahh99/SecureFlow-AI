"""secureflow.core.defence.noise — Gaussian noise and gradient clipping.

These are the two primitives that together provide (ε, δ)-differential
privacy for federated gradient updates:

  1. Gradient clipping bounds the L∞ norm of any single update,
     limiting how much one transaction can influence the model.
  2. Gaussian noise calibrated to the clipping bound provides the
     mathematical privacy guarantee.

Production configuration (Sierra Leone pilot):
  noise_std  = 0.05  (DP noise std relative to the clipping bound)
  clip_value = 0.1   (gradient L∞ clipping threshold)
"""

from __future__ import annotations

import numpy as np


def add_gaussian_noise(parameters: list[np.ndarray],
                       noise_std: float = 0.05) -> list[np.ndarray]:
    """Add calibrated Gaussian noise to gradient parameters.

    Args:
        parameters: list of NumPy arrays (Flower exchange format)
        noise_std:  standard deviation of the noise

    Returns:
        New list of arrays with noise added.
    """
    return [
        p + np.random.normal(0, noise_std, p.shape).astype(p.dtype)
        for p in parameters
    ]


def clip_gradients(parameters: list[np.ndarray],
                   clip_value: float = 0.1) -> list[np.ndarray]:
    """Clip gradient values element-wise to [-clip_value, clip_value].

    Args:
        parameters: list of NumPy arrays
        clip_value: clipping threshold

    Returns:
        New list of arrays with values clipped.
    """
    return [np.clip(p, -clip_value, clip_value) for p in parameters]


def apply_differential_privacy(parameters: list[np.ndarray],
                                noise_std: float = 0.05,
                                clip_value: float = 0.1) -> list[np.ndarray]:
    """Apply the full DP step: clip then add noise.

    This is what makes the gradients safe to share with the federation
    server. Without this, gradient updates can be inverted to reconstruct
    individual training transactions.
    """
    clipped = clip_gradients(parameters, clip_value)
    return add_gaussian_noise(clipped, noise_std)
