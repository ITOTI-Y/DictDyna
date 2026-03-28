"""Shared utilities for environment modules."""

import numpy as np


def rbc_midpoint_action(env) -> np.ndarray:
    """Simple rule-based control action (midpoint of action space)."""
    return (env.action_space.low + env.action_space.high) / 2.0
