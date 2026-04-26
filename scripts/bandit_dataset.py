"""
Canonical bandit dataset entrypoints for the real-data vehicle experiments.

This module is a thin organizing layer over the existing builder files so that
new code can import one stable API without breaking the older notebooks.
"""

from typing import Any

try:
    from .bandit_dataloader import build_bandit_dataset as _build_triggered_dataset
    from .bandit_dataloader_simple import (
        build_bandit_dataset_simple as _build_full_universe_gaussian_dataset,
    )
    from .bandit_dataloader_utility import (
        build_bandit_dataset_utility as _build_full_universe_rational_dataset,
    )
    from .bandit_dataloader_utility_pc_linear import (
        build_bandit_dataset_utility_pc_linear as _build_full_universe_piecewise_linear_dataset,
    )
except ImportError:
    from bandit_dataloader import build_bandit_dataset as _build_triggered_dataset
    from bandit_dataloader_simple import (
        build_bandit_dataset_simple as _build_full_universe_gaussian_dataset,
    )
    from bandit_dataloader_utility import (
        build_bandit_dataset_utility as _build_full_universe_rational_dataset,
    )
    from bandit_dataloader_utility_pc_linear import (
        build_bandit_dataset_utility_pc_linear as _build_full_universe_piecewise_linear_dataset,
    )


def build_triggered_bandit_dataset(*args: Any, **kwargs: Any):
    return _build_triggered_dataset(*args, **kwargs)


def build_full_universe_gaussian_dataset(*args: Any, **kwargs: Any):
    return _build_full_universe_gaussian_dataset(*args, **kwargs)


def build_full_universe_rational_dataset(*args: Any, **kwargs: Any):
    return _build_full_universe_rational_dataset(*args, **kwargs)


def build_full_universe_piecewise_linear_dataset(*args: Any, **kwargs: Any):
    return _build_full_universe_piecewise_linear_dataset(*args, **kwargs)


def build_bandit_dataset_variant(*args: Any, variant: str = "rational", **kwargs: Any):
    """
    Dispatch to one of the existing bandit dataset builders.

    Supported variants:
    - "triggered"
    - "gaussian"
    - "rational"
    - "piecewise_linear"
    - "pc_linear"
    """
    variant_key = variant.strip().lower()

    if variant_key == "triggered":
        return build_triggered_bandit_dataset(*args, **kwargs)
    if variant_key == "gaussian":
        return build_full_universe_gaussian_dataset(*args, **kwargs)
    if variant_key == "rational":
        return build_full_universe_rational_dataset(*args, **kwargs)
    if variant_key in {"piecewise_linear", "pc_linear"}:
        return build_full_universe_piecewise_linear_dataset(*args, **kwargs)

    raise ValueError(
        "Unknown variant "
        f"{variant!r}. Expected one of: "
        "'triggered', 'gaussian', 'rational', 'piecewise_linear', 'pc_linear'."
    )
