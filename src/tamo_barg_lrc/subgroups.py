"""Finite-field additive and multiplicative subgroup and coset utilities."""

from collections.abc import Iterable, Sequence
from itertools import combinations
from typing import Any, Literal

from sage.arith.misc import divisors
from sage.modules.free_module import VectorSpace
from sage.rings.finite_rings.finite_field_constructor import GF

SubgroupType = Literal["add", "mult"]

__all__ = [
    "SubgroupType",
    "field_element_to_int",
    "sort_field_elements",
    "is_closed_under_addition",
    "find_subspaces",
    "find_additive_subgroups",
    "find_additive_subgroups_bruteforce",
    "list_multiplicative_subgroups",
    "list_canonical_additive_subgroups",
    "shift_additive_subgroup",
    "list_subgroup_cosets",
]


def _validate_finite_field(field: Any) -> None:
    """Raise ``ValueError`` if ``field`` is not a finite field."""
    if not hasattr(field, "is_finite") or not field.is_finite():
        raise ValueError("field must be a finite field")


def field_element_to_int(elem: Any) -> int:
    """Map a finite-field element deterministically to an integer in ``0 .. q - 1``.

    Args:
        elem: An element of a prime or extension finite field (or integer).

    Returns:
        Deterministic non-negative integer representation with ``0`` mapped to ``0``.
    """
    if hasattr(elem, "is_zero") and elem.is_zero():
        return 0
    if hasattr(elem, "to_integer"):
        return int(elem.to_integer())
    return int(elem)


def sort_field_elements(elements: Iterable[Any]) -> list[Any]:
    """Return finite-field elements sorted deterministically by integer representation.

    Args:
        elements: Iterable of finite-field elements.

    Returns:
        Sorted list of elements.
    """
    return sorted(elements, key=field_element_to_int)


def is_closed_under_addition(subgroup: Iterable[Any]) -> bool:
    """Check whether a non-empty subset of a finite field is closed under addition.

    Args:
        subgroup: Candidate subset of finite-field elements.

    Returns:
        ``True`` if ``subgroup`` is non-empty, contains ``0``, and is closed under
        pairwise addition; ``False`` otherwise.
    """
    elem_set = set(subgroup)
    if not elem_set:
        return False

    first = next(iter(elem_set))
    zero = first.parent().zero() if hasattr(first, "parent") else type(first)(0)
    if zero not in elem_set:
        return False

    elems = list(elem_set)
    n = len(elems)
    for i in range(n):
        a = elems[i]
        for j in range(i, n):
            if (a + elems[j]) not in elem_set:
                return False
    return True


def find_subspaces(dim: int, ambient_dim: int, base_field: Any) -> list[Any]:
    """Enumerate all ``dim``-dimensional subspaces of ``GF(p)^ambient_dim``.

    Args:
        dim: Target subspace dimension.
        ambient_dim: Ambient vector-space dimension over the prime subfield.
        base_field: Finite field whose characteristic defines the prime subfield.

    Returns:
        List of all ``dim``-dimensional subspaces of ``GF(p)^ambient_dim``.

    Raises:
        ValueError: If ``base_field`` is not finite or ``dim`` is outside
            ``0 .. ambient_dim``.
    """
    _validate_finite_field(base_field)
    dim_int = int(dim)
    ambient_int = int(ambient_dim)
    if ambient_int < 0 or not (0 <= dim_int <= ambient_int):
        raise ValueError(f"Subspace dimension {dim_int} must satisfy 0 <= dim <= {ambient_int}")

    prime_field = GF(base_field.characteristic())
    ambient_space = VectorSpace(prime_field, ambient_int)
    return list(ambient_space.subspaces(dim_int))


def find_additive_subgroups(field: Any, size: int) -> list[list[Any]]:
    """Find all additive subgroups of ``field`` of the given ``size``.

    Args:
        field: Finite field ``GF(p^m)``.
        size: Desired subgroup cardinality (must be a power ``p^t`` dividing ``p^m``).

    Returns:
        Deterministic list of sorted additive subgroups of cardinality ``size``.

    Raises:
        ValueError: If ``field`` is not finite or ``size`` is not a power of the
            field characteristic dividing ``field.cardinality()``.
    """
    _validate_finite_field(field)
    size_int = int(size)
    p = int(field.characteristic())
    m = int(field.degree())
    q = int(field.cardinality())

    if size_int < 1:
        raise ValueError(
            f"Additive subgroup size {size_int} must be a power of characteristic {p} dividing {q}"
        )

    dim = 0
    rem = size_int
    while rem > 1:
        if rem % p != 0:
            raise ValueError(
                f"Additive subgroup size {size_int} must be a power of "
                f"characteristic {p} dividing {q}"
            )
        rem //= p
        dim += 1

    if dim > m:
        raise ValueError(
            f"Additive subgroup size {size_int} must be a power of characteristic {p} dividing {q}"
        )

    alpha = field.gen()
    zero = field.zero()
    field_basis = [alpha**j for j in range(m)]

    subgroups: list[list[Any]] = []
    for subspace in find_subspaces(dim, m, field):
        subgroup = [
            sum(
                (field(int(coord)) * field_basis[j] for j, coord in enumerate(vec)),
                zero,
            )
            for vec in subspace
        ]
        subgroups.append(sort_field_elements(subgroup))

    subgroups.sort(key=lambda sg: [field_element_to_int(x) for x in sg])
    return subgroups


def find_additive_subgroups_bruteforce(
    field: Any, size: int, max_subgroups: int = 1000
) -> list[list[Any]]:
    """Search for additive subgroups of ``field`` of cardinality ``size`` by combinations.

    Iterates lazily over ``(size - 1)``-subsets of non-zero field elements together
    with ``0``, avoiding materializing all combinations in memory.

    Args:
        field: Finite field ``GF(q)``.
        size: Desired subgroup cardinality dividing ``q``.
        max_subgroups: Maximum number of subgroups to collect before stopping.

    Returns:
        Deterministic list of up to ``max_subgroups`` sorted additive subgroups.

    Raises:
        ValueError: If ``field`` is not finite, ``size < 1``, ``q % size != 0``,
            or ``max_subgroups < 1``.
    """
    _validate_finite_field(field)
    size_int = int(size)
    max_count = int(max_subgroups)
    q = int(field.cardinality())

    if size_int < 1 or q % size_int != 0:
        raise ValueError(f"Additive subgroup size {size_int} must be a positive divisor of {q}")
    if max_count < 1:
        raise ValueError(f"max_subgroups ({max_count}) must be at least 1")

    zero = field.zero()
    non_zero_elements = [x for x in sort_field_elements(field) if not x.is_zero()]

    subgroups: list[list[Any]] = []
    for combo in combinations(non_zero_elements, size_int - 1):
        candidate = [zero, *combo]
        if is_closed_under_addition(candidate):
            subgroups.append(sort_field_elements(candidate))
            if len(subgroups) >= max_count:
                break

    subgroups.sort(key=lambda sg: [field_element_to_int(x) for x in sg])
    return subgroups


def list_multiplicative_subgroups(field: Any) -> dict[int, list[Any]]:
    """Construct all multiplicative subgroups of ``field^*`` keyed by subgroup order.

    Args:
        field: Finite field ``GF(q)``.

    Returns:
        Dictionary mapping each divisor ``d`` of ``q - 1`` to the unique sorted
        multiplicative subgroup of order ``d``.

    Raises:
        ValueError: If ``field`` is not finite.
    """
    _validate_finite_field(field)
    q_minus_1 = int(field.cardinality()) - 1
    gen = field.multiplicative_generator()
    one = field.one()

    subgroups: dict[int, list[Any]] = {}
    for d in divisors(q_minus_1):
        d_int = int(d)
        step = gen ** (q_minus_1 // d_int)
        elems = [one]
        curr = one
        for _ in range(d_int - 1):
            curr = curr * step
            elems.append(curr)
        subgroups[d_int] = sort_field_elements(elems)

    return subgroups


def list_canonical_additive_subgroups(field: Any) -> dict[int, list[Any]]:
    """Construct canonical nested ``GF(p)``-subspace additive subgroups of ``field``.

    Builds ``span(1, alpha, ..., alpha^(i - 1))`` for all ``i`` in ``0 .. deg(field)``,
    including the trivial subgroup ``{0}`` of size ``1`` and the full field of size ``q``.

    Args:
        field: Finite field ``GF(p^m)``.

    Returns:
        Dictionary mapping each size ``p^i`` (``0 <= i <= m``) to its sorted
        canonical additive subgroup.

    Raises:
        ValueError: If ``field`` is not finite.
    """
    _validate_finite_field(field)
    p = int(field.characteristic())
    m = int(field.degree())
    alpha = field.gen()

    current = [field.zero()]
    result: dict[int, list[Any]] = {1: list(current)}
    for i in range(m):
        basis_elem = alpha**i
        current = [h + c * basis_elem for c in range(p) for h in current]
        result[len(current)] = sort_field_elements(current)

    return result


def shift_additive_subgroup(field: Any, subgroup: Sequence[Any], shift_exponent: int) -> list[Any]:
    """Scale an additive subgroup by ``g^shift_exponent`` to obtain a shifted subspace.

    Args:
        field: Finite field ``GF(q)``.
        subgroup: Sequence of elements forming an additive subgroup of ``field``.
        shift_exponent: Integer power of the primitive element to multiply by.

    Returns:
        Sorted list of elements in ``(g^shift_exponent) * subgroup``.

    Raises:
        ValueError: If ``field`` is not finite.
    """
    _validate_finite_field(field)
    gen = (
        field.primitive_element()
        if hasattr(field, "primitive_element") and field.degree() > 1
        else field.multiplicative_generator()
    )
    multiplier = gen ** int(shift_exponent)
    return sort_field_elements(field(h) * multiplier for h in subgroup)


def list_subgroup_cosets(
    field: Any, subgroup: Sequence[Any], subgroup_type: SubgroupType
) -> list[list[Any]]:
    """Partition ``field`` (or ``field^*``) into deterministic cosets of ``subgroup``.

    Args:
        field: Finite field ``GF(q)``.
        subgroup: Non-empty additive or multiplicative subgroup of ``field``.
        subgroup_type: ``"add"`` for additive cosets over ``GF(q)``, or ``"mult"``
            for multiplicative cosets over ``GF(q)^*``.

    Returns:
        Deterministically ordered list of sorted cosets, computed in ``O(q)``
        field operations.

    Raises:
        ValueError: If ``field`` is not finite, ``subgroup`` is empty,
            ``subgroup_type`` is invalid, or a multiplicative subgroup contains ``0``.
    """
    _validate_finite_field(field)
    if subgroup_type not in ("add", "mult"):
        raise ValueError(f"subgroup_type must be 'add' or 'mult', got {subgroup_type!r}")
    if len(subgroup) < 1:
        raise ValueError("subgroup must be non-empty")

    sg_elems = [field(h) for h in subgroup]
    if subgroup_type == "mult" and any(h.is_zero() for h in sg_elems):
        raise ValueError("Multiplicative subgroup cannot contain zero")

    ordered_field = sort_field_elements(field)
    candidates = (
        ordered_field if subgroup_type == "add" else [x for x in ordered_field if not x.is_zero()]
    )

    visited: set[Any] = set()
    cosets: list[list[Any]] = []
    for g in candidates:
        if g in visited:
            continue
        coset = sort_field_elements(h + g if subgroup_type == "add" else h * g for h in sg_elems)
        visited.update(coset)
        cosets.append(coset)

    return cosets
