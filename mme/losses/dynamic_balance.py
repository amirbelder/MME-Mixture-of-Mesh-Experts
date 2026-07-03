"""Dynamic balancing between task, diversity, and similarity losses.

Users pass a schedule ``(t, total_steps) -> (alpha, beta)`` or one of the
provided helpers.
"""

from __future__ import annotations

import math
from typing import Callable, Sequence, Tuple


Schedule = Callable[[int, int], Tuple[float, float]]


def linear_schedule(
    alpha_start: float = 0.0,
    alpha_end: float = 0.1,
    beta_start: float = 0.0,
    beta_end: float = 0.1,
) -> Schedule:
    def f(t: int, total: int) -> Tuple[float, float]:
        r = min(1.0, max(0.0, t / max(1, total - 1)))
        return (
            alpha_start + (alpha_end - alpha_start) * r,
            beta_start + (beta_end - beta_start) * r,
        )

    return f


def cosine_schedule(
    alpha_start: float = 0.0,
    alpha_end: float = 0.1,
    beta_start: float = 0.0,
    beta_end: float = 0.1,
) -> Schedule:
    def f(t: int, total: int) -> Tuple[float, float]:
        r = min(1.0, max(0.0, t / max(1, total - 1)))
        cos = 0.5 * (1.0 - math.cos(math.pi * r))
        return (
            alpha_start + (alpha_end - alpha_start) * cos,
            beta_start + (beta_end - beta_start) * cos,
        )

    return f


def step_schedule(steps: Sequence[Tuple[int, float, float]]) -> Schedule:
    """``steps`` = list of ``(after_step, alpha, beta)`` breakpoints (ascending)."""
    sorted_steps = sorted(steps)

    def f(t: int, total: int) -> Tuple[float, float]:  # noqa: ARG001
        a, b = 0.0, 0.0
        for after, aa, bb in sorted_steps:
            if t >= after:
                a, b = aa, bb
        return a, b

    return f


class DynamicBalancedLoss:
    """Combines task + alpha*diversity + beta*similarity with a schedule.

    Args:
        task_loss_fn: ``(logits, targets) -> torch.Tensor``.
        diversity_loss_fn: ``(expert_features) -> torch.Tensor``.
        similarity_loss_fn: ``(expert_logits, moe_logits) -> torch.Tensor``.
        schedule: ``(t, total_steps) -> (alpha, beta)``.
    """

    def __init__(
        self,
        task_loss_fn,
        diversity_loss_fn,
        similarity_loss_fn,
        schedule: Schedule,
    ) -> None:
        self.task = task_loss_fn
        self.div = diversity_loss_fn
        self.sim = similarity_loss_fn
        self.schedule = schedule

    def __call__(
        self,
        logits,
        targets,
        expert_features: Sequence,
        expert_logits: Sequence,
        step: int,
        total_steps: int,
    ):
        alpha, beta = self.schedule(step, total_steps)
        l_task = self.task(logits, targets)
        l_div = self.div(expert_features) if expert_features else 0.0
        l_sim = self.sim(expert_logits, logits) if expert_logits else 0.0
        total = l_task + alpha * l_div + beta * l_sim
        return total, {
            "task": float(l_task),
            "div": float(l_div) if hasattr(l_div, "__float__") else float(l_div),
            "sim": float(l_sim) if hasattr(l_sim, "__float__") else float(l_sim),
            "alpha": alpha,
            "beta": beta,
        }
