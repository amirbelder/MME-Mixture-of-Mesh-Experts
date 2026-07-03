"""Losses: task, diversity, similarity, and dynamic balance."""

from mme.losses.diversity import diversity_loss
from mme.losses.dynamic_balance import (
    cosine_schedule,
    DynamicBalancedLoss,
    linear_schedule,
    step_schedule,
)
from mme.losses.similarity import similarity_loss
from mme.losses.task import task_ce_loss

__all__ = [
    "task_ce_loss",
    "diversity_loss",
    "similarity_loss",
    "DynamicBalancedLoss",
    "linear_schedule",
    "cosine_schedule",
    "step_schedule",
]
