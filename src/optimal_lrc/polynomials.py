"""Univariate polynomial algebra utilities for Tamo-Barg encoding and erasure recovery."""

from collections.abc import Sequence
from typing import Any, Literal

from sage.matrix.constructor import matrix
from sage.misc.misc_c import prod

from optimal_lrc.subgroups import field_element_to_int

__all__ = [
    "build_annihilator_polynomial",
    "build_good_polynomial",
    "build_algebra_basis",
    "build_encoding_basis",
    "intersect_encoding_bases",
    "evaluate_message_polynomial",
    "interpolate_erasures",
]


def build_annihilator_polynomial(poly_ring: Any, evaluation_points: Sequence[Any]) -> Any:
    """Construct the monic vanishing polynomial ``prod_{pt}(x - pt)`` via a product tree.

    Args:
        poly_ring: Univariate polynomial ring ``GF(q)['x']``.
        evaluation_points: Sequence of evaluation points in ``GF(q)``.

    Returns:
        Monic polynomial in ``poly_ring`` vanishing on ``evaluation_points``.
    """
    x = poly_ring.gen()
    base_ring = poly_ring.base_ring()
    return prod([x - base_ring(pt) for pt in evaluation_points], poly_ring.one())


def build_good_polynomial(
    poly_ring: Any,
    subgroup: Sequence[Any],
    subgroup_type: Literal["add", "mult"],
    annihilator: Any | None = None,
) -> Any:
    """Construct a Tamo-Barg good polynomial constant on cosets of ``subgroup``.

    Args:
        poly_ring: Univariate polynomial ring ``GF(q)['x']``.
        subgroup: Non-empty additive or multiplicative subgroup of ``GF(q)``.
        subgroup_type: ``"add"`` (uses ``prod_{h in H}(x - h)``) or ``"mult"``
            (uses ``x^|H|``).
        annihilator: Optional vanishing polynomial used to reduce degree if needed.

    Returns:
        Good polynomial in ``poly_ring``.

    Raises:
        ValueError: If ``subgroup_type`` is not ``"add"`` or ``"mult"``, or if
            ``subgroup`` is empty.
    """
    if subgroup_type not in ("add", "mult"):
        raise ValueError(f"subgroup_type must be 'add' or 'mult', got {subgroup_type!r}")
    if len(subgroup) < 1:
        raise ValueError("subgroup must be non-empty")

    x = poly_ring.gen()
    base_ring = poly_ring.base_ring()
    if subgroup_type == "mult":
        good = x ** len(subgroup)
    else:
        good = prod([x - base_ring(h) for h in subgroup], poly_ring.one())

    if annihilator is not None and good.degree() >= annihilator.degree():
        good = good % annihilator
    return good


def build_algebra_basis(
    good_poly: Any, partition_size: int, annihilator: Any | None = None
) -> list[Any]:
    """Generate powers ``[1, g, g^2, ..., g^(partition_size - 1)]`` of a good polynomial.

    Args:
        good_poly: Good polynomial ``g(x)`` in ``GF(q)['x']``.
        partition_size: Number of coset blocks in the partition.
        annihilator: Optional vanishing polynomial used for modular reduction.

    Returns:
        List of ``partition_size`` powers of ``good_poly``.

    Raises:
        ValueError: If ``partition_size < 0``.
    """
    count = int(partition_size)
    if count < 0:
        raise ValueError(f"partition_size ({count}) must be non-negative")
    if count == 0:
        return []

    poly_ring = good_poly.parent()
    curr = poly_ring.one()
    if annihilator is not None and curr.degree() >= annihilator.degree():
        curr = curr % annihilator

    basis = [curr]
    for _ in range(1, count):
        curr = curr * good_poly
        if annihilator is not None and curr.degree() >= annihilator.degree():
            curr = curr % annihilator
        basis.append(curr)
    return basis


def build_encoding_basis(
    poly_ring: Any,
    algebra_basis: Sequence[Any],
    locality: int,
    annihilator: Any | None = None,
) -> list[Any]:
    """Construct the Tamo-Barg encoding basis ``{x^i * g(x)^j}`` sorted by degree.

    Args:
        poly_ring: Univariate polynomial ring ``GF(q)['x']``.
        algebra_basis: Powers ``[1, g, ..., g^(s - 1)]`` of the good polynomial.
        locality: Locality parameter ``r`` (exponents ``0 <= i < r``).
        annihilator: Optional vanishing polynomial used for modular reduction.

    Returns:
        Deterministically sorted list of ``locality * len(algebra_basis)`` basis
        polynomials ordered primarily by ascending degree.

    Raises:
        ValueError: If ``locality < 0``.
    """
    r_int = int(locality)
    if r_int < 0:
        raise ValueError(f"locality ({r_int}) must be non-negative")
    if r_int == 0 or not algebra_basis:
        return []

    enc_basis: list[Any] = []
    for shift_deg in range(r_int):
        for p in algebra_basis:
            prod_poly = p.shift(shift_deg) if shift_deg > 0 else p
            if annihilator is not None and prod_poly.degree() >= annihilator.degree():
                prod_poly = prod_poly % annihilator
            enc_basis.append(prod_poly)

    enc_basis.sort(key=lambda poly: poly.degree())
    if any(enc_basis[i].degree() == enc_basis[i + 1].degree() for i in range(len(enc_basis) - 1)):
        enc_basis.sort(
            key=lambda poly: (
                poly.degree(),
                [field_element_to_int(c) for c in reversed(poly.list())],
            )
        )
    return enc_basis


def intersect_encoding_bases(
    poly_ring: Any,
    basis_a: Sequence[Any],
    basis_b: Sequence[Any],
    length: int,
) -> list[Any]:
    """Compute a degree-minimizing basis of ``span(basis_a) & span(basis_b)``.

    Uses Zassenhaus subspace intersection followed by reverse-column reduced row
    echelon form so pivots eliminate high-degree terms first, yielding basis
    polynomials of strictly ascending, minimal degrees.

    Args:
        poly_ring: Univariate polynomial ring ``GF(q)['x']``.
        basis_a: First polynomial basis sequence.
        basis_b: Second polynomial basis sequence.
        length: Code length upper-bounding polynomial degrees (``deg < length``).

    Returns:
        Basis of ``span(basis_a) & span(basis_b)`` in ``GF(q)[x]_{< length}``,
        sorted in strictly ascending polynomial degree.

    Raises:
        ValueError: If ``length < 0``.
    """
    n_len = int(length)
    if n_len < 0:
        raise ValueError(f"length ({n_len}) must be non-negative")
    if n_len == 0 or not basis_a or not basis_b:
        return []

    base_field = poly_ring.base_ring()
    num_cols = max(
        n_len,
        max((int(p.degree()) + 1 for p in basis_a), default=0),
        max((int(p.degree()) + 1 for p in basis_b), default=0),
    )

    mat_a = matrix(
        base_field,
        len(basis_a),
        num_cols,
        [p.padded_list(num_cols) for p in basis_a],
    )
    mat_b = matrix(
        base_field,
        len(basis_b),
        num_cols,
        [p.padded_list(num_cols) for p in basis_b],
    )

    ker = mat_a.stack(-mat_b).left_kernel_matrix()
    if ker.nrows() == 0:
        return []

    basis_mat = ker.matrix_from_columns(list(range(mat_a.nrows()))) * mat_a
    rev_cols = list(range(num_cols - 1, -1, -1))
    rref_rev = basis_mat.matrix_from_columns(rev_cols).echelon_form()

    last_col = num_cols - 1
    polys: list[Any] = []
    for r_rev in rref_rev:
        nz = r_rev.nonzero_positions()
        if not nz:
            continue
        deg = last_col - nz[0]
        if deg < n_len:
            coeffs: list[Any] = [0] * (deg + 1)
            for idx in nz:
                coeffs[last_col - idx] = r_rev[idx]
            polys.append(poly_ring(coeffs))

    polys.sort(key=lambda p: p.degree())
    return polys


def evaluate_message_polynomial(
    poly_ring: Any, basis: Sequence[Any], message: Sequence[Any]
) -> Any:
    """Form the message encoding polynomial ``sum_i message[i] * basis[i]``.

    Args:
        poly_ring: Univariate polynomial ring ``GF(q)['x']``.
        basis: Encoding polynomial basis of length ``k``.
        message: Message vector or sequence of length ``k`` over ``GF(q)``.

    Returns:
        Encoding polynomial in ``poly_ring``.

    Raises:
        ValueError: If ``len(message) != len(basis)``.
    """
    if len(message) != len(basis):
        raise ValueError(f"Message length ({len(message)}) must match basis length ({len(basis)})")
    base_ring = poly_ring.base_ring()
    return sum(
        (base_ring(message[i]) * basis[i] for i in range(len(basis))),
        poly_ring.zero(),
    )


def interpolate_erasures(
    poly_ring: Any,
    eval_points: Sequence[Any],
    values: Sequence[Any],
    erasure_mask: Sequence[Any],
) -> list[Any]:
    """Recover erased evaluations via univariate Lagrange interpolation.

    Args:
        poly_ring: Univariate polynomial ring ``GF(q)['x']``.
        eval_points: Evaluation points ``[alpha_0, ..., alpha_{m - 1}]``.
        values: Received symbol values aligned with ``eval_points``.
        erasure_mask: Binary/integer erasure indicators (``0`` = known, non-zero = erased).

    Returns:
        New list of values with erased positions replaced by interpolated evaluations.

    Raises:
        ValueError: If ``eval_points``, ``values``, and ``erasure_mask`` differ in length.
    """
    n = len(eval_points)
    if len(values) != n or len(erasure_mask) != n:
        raise ValueError("eval_points, values, and erasure_mask must all have the same length")

    known_pts: list[tuple[Any, Any]] = []
    erased_indices: list[int] = []
    for i in range(n):
        if int(erasure_mask[i]) == 0:
            known_pts.append((eval_points[i], values[i]))
        else:
            erased_indices.append(i)

    out = list(values)
    if not erased_indices:
        return out

    interp_poly = poly_ring.lagrange_polynomial(known_pts)
    for idx in erased_indices:
        out[idx] = interp_poly(eval_points[idx])
    return out
