"""Post-processing utilities for error-weight distribution simulation CSV outputs."""

import csv
from pathlib import Path
from typing import Any

__all__ = ["process_error_distribution_csv"]


def process_error_distribution_csv(
    err_csv_path: str | Path,
    design_distance: int,
    real_distance: int,
    sim_itr: int = 5000,
    threshold_pct: float = 99.0,
) -> dict[str, Any]:
    """Compute post-global-decoding success probabilities from an error distribution CSV.

    Counts trials whose remaining error weight after iterative local decoding does
    not exceed the global decoder's unique-decoding radius
    ``(design_distance - 1) // 2``.

    Args:
        err_csv_path: Path to a ``*_err.csv`` file with columns
            ``[Num_of_error_symbols, remain_err_weight, count, dist]``.
        design_distance: Polynomial design distance of the code (``>= 1``).
        real_distance: Number of error-weight bins (``1 .. real_distance``) in the
            returned ``succ_count`` and ``prob`` lists (``>= 1``).
        sim_itr: Number of Monte Carlo trials per error weight (``>= 1``).
        threshold_pct: Success-probability percentage threshold used to identify
            the first 0-based index falling strictly below ``threshold_pct``.

    Returns:
        Dictionary with keys:
            - ``"succ_count"``: ``list[int]`` of length ``real_distance``.
            - ``"prob"``: ``list[float]`` of success percentages in ``[0.0, 100.0]``.
            - ``"threshold_index"``: first 0-based index ``i`` with
              ``prob[i] < threshold_pct``, or ``None`` if all entries are
              ``>= threshold_pct``.

    Raises:
        ValueError: If ``design_distance < 1``, ``real_distance < 1``, or
            ``sim_itr < 1``.
    """
    if design_distance < 1:
        raise ValueError(f"design_distance must be >= 1, got {design_distance}")
    if real_distance < 1:
        raise ValueError(f"real_distance must be >= 1, got {real_distance}")
    if sim_itr < 1:
        raise ValueError(f"sim_itr must be >= 1, got {sim_itr}")

    max_correctable = (design_distance - 1) // 2
    succ_count: list[int] = [0] * real_distance

    with open(err_csv_path, newline="") as csvfile:
        err_reader = csv.reader(csvfile)
        for row in err_reader:
            if len(row) < 3 or row[1] == "remain_err_weight" or row[0] == "Num_of_error_symbols":
                continue
            err_idx = int(row[0]) - 1
            remain_wt = int(row[1])
            count = int(row[2])
            if 0 <= err_idx < real_distance and remain_wt <= max_correctable:
                succ_count[err_idx] += count

    prob: list[float] = [(count / sim_itr) * 100.0 for count in succ_count]

    threshold_index: int | None = None
    for i, pct in enumerate(prob):
        if pct < threshold_pct and threshold_index is None:
            threshold_index = i
            break

    return {
        "succ_count": succ_count,
        "prob": prob,
        "threshold_index": threshold_index,
    }
