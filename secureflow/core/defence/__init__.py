"""secureflow.core.defence — Differential privacy mechanisms.

Provides the cryptographic primitives that protect operator data
before gradient updates leave the operator's infrastructure.
"""

from secureflow.core.defence.noise import (
    add_gaussian_noise,
    clip_gradients,
    apply_differential_privacy,
)

__all__ = ["add_gaussian_noise", "clip_gradients", "apply_differential_privacy"]
