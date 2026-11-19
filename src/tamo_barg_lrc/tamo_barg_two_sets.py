"""Two-set Tamo-Barg locally recoverable codes, vector encoder, and decoders."""

import numbers
from collections.abc import Sequence
from copy import copy
from typing import Any, Literal

import networkx as nx
from sage.categories.cartesian_product import cartesian_product
from sage.coding.decoder import Decoder, DecodingError
from sage.coding.encoder import Encoder
from sage.coding.grs_code import GeneralizedReedSolomonCode
from sage.coding.linear_code import AbstractLinearCode
from sage.matrix.constructor import matrix
from sage.misc.cachefunc import cached_method
from sage.modules.free_module import VectorSpace
from sage.modules.free_module_element import vector
from sage.rings.finite_rings.finite_field_constructor import GF
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing

from tamo_barg_lrc.expander import create_bipartite_coset_graph
from tamo_barg_lrc.polynomials import (
    build_algebra_basis,
    build_encoding_basis,
    build_good_polynomial,
    evaluate_message_polynomial,
    interpolate_erasures,
    intersect_encoding_bases,
)
from tamo_barg_lrc.subgroups import (
    SubgroupType,
    is_closed_under_addition,
    list_canonical_additive_subgroups,
    list_multiplicative_subgroups,
    list_subgroup_cosets,
    shift_additive_subgroup,
    sort_field_elements,
)

__all__ = [
    "TamoBargCodeTwoSets",
    "TamoBargGlobalDecoder",
    "TamoBargGlobalEEDecoder",
    "TamoBargIterativeDecoder",
    "TamoBargIterativeEEDecoder",
    "TamoBargTwoSetsVectorEncoder",
    "TamoBargTwoStepDecoder",
    "TamoBargTwoStepEEDecoder",
]


def _is_int_like(value: Any) -> bool:
    """Return True if ``value`` is an integer (Python ``int`` or Sage ``Integer``), not ``bool``."""
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def _int_log_base(size: int, base: int) -> int:
    """Compute exact integer ``log_{base}(size)`` for a prime-power subgroup size."""
    exp = 0
    rem = size
    while rem > 1:
        if rem % base != 0:
            raise ValueError(f"Subgroup size {size} is not a power of characteristic {base}")
        rem //= base
        exp += 1
    return exp


def _safe_parity_check_matrix(grs_code: GeneralizedReedSolomonCode) -> Any:
    """Return the parity-check matrix of ``grs_code``, handling trivial full-rate codes safely."""
    if grs_code.dimension() == grs_code.length():
        return matrix(grs_code.base_field(), 0, grs_code.length())
    return grs_code.parity_check_matrix()


class TamoBargCodeTwoSets(AbstractLinearCode):
    """Optimal locally recoverable Tamo-Barg code with two disjoint-repair-set partitions.

    Constructs evaluation codes over ``GF(q)`` partitioned simultaneously by two
    (additive or multiplicative) subgroups ``H_0`` and ``H_1``, enabling iterative,
    global, and two-step error and erasure-and-error decoding.
    """

    _registered_encoders: dict[str, type[Encoder]] = {}
    _registered_decoders: dict[str, type[Decoder]] = {}

    def __init__(
        self,
        base_field: Any,
        length: int,
        dimension: int,
        locality: Sequence[int],
        local_minimum_distance: Sequence[int] | int = (2, 2),
        sub_group_type: Sequence[Literal["add", "mult"]] = ("add", "add"),
        shift_add: bool = False,
        subgroup: Sequence[Sequence[Any] | None] | None = None,
    ) -> None:
        """Initialize a two-set Tamo-Barg LRC.

        Args:
            base_field: Finite field ``GF(q)`` over which the code is defined.
            length: Codeword length ``n``.
            dimension: Message dimension ``k`` (``1 <= k <= max_dimension()``).
            locality: Two-element sequence ``[r_0, r_1]`` of partition localities.
            local_minimum_distance: Scalar ``d`` (expanded to ``[d, d]``) or
                two-element sequence ``[d_0, d_1]`` of local minimum distances (``>= 2``).
            sub_group_type: Two-element sequence of ``"add"`` or ``"mult"`` specifying
                each partition's subgroup structure.
            shift_add: Whether to scale default additive subgroups by a primitive
                element power to alter coset alignment.
            subgroup: Optional two-element sequence ``[sg_0, sg_1]`` overriding default
                subgroup selection where entries are non-``None``.

        Raises:
            ValueError: If ``base_field`` is not finite, parameters have invalid shapes
                or ranges, required subgroups do not exist, coset sizes do not divide
                the partition domain, or ``dimension`` exceeds ``max_dimension()``.
        """
        if not hasattr(base_field, "is_finite") or not base_field.is_finite():
            raise ValueError("base_field must be a finite field")

        if not _is_int_like(length) or int(length) < 1:
            raise ValueError(f"length must be a positive integer, got {length!r}")
        if not _is_int_like(dimension) or int(dimension) < 1:
            raise ValueError(f"dimension must be a positive integer, got {dimension!r}")
        length_int = int(length)
        dim_int = int(dimension)

        if not isinstance(locality, (list, tuple)) or len(locality) != 2:
            raise ValueError("locality must be a 2-element sequence [r1, r2]")
        if not (_is_int_like(locality[0]) and _is_int_like(locality[1])):
            raise ValueError(f"locality elements must be integers, got {locality!r}")
        self._num_of_sets: int = 2
        self._locality: list[int] = [int(locality[0]), int(locality[1])]
        if any(r_val < 1 for r_val in self._locality):
            raise ValueError(f"locality values must be >= 1, got {self._locality}")

        if _is_int_like(local_minimum_distance):
            d_val = int(local_minimum_distance)
            self._local_minimum_distance: list[int] = [d_val, d_val]
        elif isinstance(local_minimum_distance, (list, tuple)) and len(local_minimum_distance) == 2:
            if not (
                _is_int_like(local_minimum_distance[0]) and _is_int_like(local_minimum_distance[1])
            ):
                raise ValueError(
                    "local_minimum_distance elements must be integers, "
                    f"got {local_minimum_distance!r}"
                )
            self._local_minimum_distance = [
                int(local_minimum_distance[0]),
                int(local_minimum_distance[1]),
            ]
        else:
            raise ValueError(
                "local_minimum_distance must be an integer or a 2-element sequence [d1, d2]"
            )
        if any(d_val < 2 for d_val in self._local_minimum_distance):
            raise ValueError(
                f"local_minimum_distance values must be >= 2, got {self._local_minimum_distance}"
            )

        if not isinstance(sub_group_type, (list, tuple)) or len(sub_group_type) != 2:
            raise ValueError("sub_group_type must be a 2-element sequence of 'add' or 'mult'")
        self._sub_group_type: list[str] = [str(sub_group_type[0]), str(sub_group_type[1])]
        if any(sgt not in ("add", "mult") for sgt in self._sub_group_type):
            raise ValueError(
                f"sub_group_type elements must be 'add' or 'mult', got {self._sub_group_type!r}"
            )

        if subgroup is None:
            norm_subgroup: list[Sequence[Any] | None] = [None, None]
        elif isinstance(subgroup, (list, tuple)) and len(subgroup) == 2:
            norm_subgroup = [subgroup[0], subgroup[1]]
        else:
            raise ValueError("subgroup must be None or a 2-element sequence [sg0, sg1]")

        coset_sizes = [self._locality[i] + self._local_minimum_distance[i] - 1 for i in range(2)]
        mult_subgroups = list_multiplicative_subgroups(base_field)
        add_subgroups = list_canonical_additive_subgroups(base_field)
        p = int(base_field.characteristic())

        self._sub_group: list[list[Any]] = []
        for i in (0, 1):
            target_size = coset_sizes[i]
            sgt = self._sub_group_type[i]
            custom_sg = norm_subgroup[i]

            if custom_sg is not None:
                if len(custom_sg) != target_size:
                    raise ValueError(
                        f"Custom subgroup {i} has size {len(custom_sg)}, expected {target_size}"
                    )
                coerced_sg = sort_field_elements(base_field(h) for h in custom_sg)
                if len(set(coerced_sg)) != target_size:
                    raise ValueError(f"Custom subgroup {i} contains duplicate elements")
                if sgt == "add":
                    if not is_closed_under_addition(coerced_sg):
                        raise ValueError(
                            f"Custom subgroup {i} is not closed under addition over {base_field}"
                        )
                else:
                    if (
                        target_size not in mult_subgroups
                        or coerced_sg != mult_subgroups[target_size]
                    ):
                        raise ValueError(
                            f"Custom subgroup {i} is not a valid multiplicative subgroup "
                            f"of order {target_size} over {base_field}"
                        )
                self._sub_group.append(coerced_sg)
            elif sgt == "mult":
                if target_size not in mult_subgroups:
                    raise ValueError(
                        f"Cannot find multiplicative subgroup of size {target_size} over "
                        f"{base_field}; available sizes: {sorted(mult_subgroups)}"
                    )
                self._sub_group.append(mult_subgroups[target_size])
            else:
                if target_size not in add_subgroups:
                    raise ValueError(
                        f"Cannot find additive subgroup of size {target_size} over "
                        f"{base_field}; available sizes: {sorted(add_subgroups)}"
                    )
                sg = add_subgroups[target_size]
                if self._sub_group_type == ["add", "add"]:
                    if i == 0:
                        if shift_add:
                            sg = shift_additive_subgroup(base_field, sg, 1)
                    else:
                        t_0 = _int_log_base(len(self._sub_group[0]), p)
                        shift_exp = (1 + t_0) if shift_add else t_0
                        sg = shift_additive_subgroup(base_field, sg, shift_exp)
                elif shift_add:
                    shift_exp = _int_log_base(len(sg), p)
                    sg = shift_additive_subgroup(base_field, sg, shift_exp)
                self._sub_group.append(sg)

        q = int(base_field.cardinality())
        has_add = "add" in self._sub_group_type
        self._partition_size: list[int] = []
        self._partition: list[list[list[Any]]] = []

        for i in (0, 1):
            sgt_lit: SubgroupType = "add" if self._sub_group_type[i] == "add" else "mult"
            c_size = coset_sizes[i]
            if sgt_lit == "add":
                if length_int > q or length_int % c_size != 0:
                    raise ValueError(
                        f"Code length ({length_int}) must be <= {q} and divisible by "
                        f"additive coset size ({c_size}) for partition {i}"
                    )
                p_size = length_int // c_size
            elif has_add and length_int == q:
                if (q - 1) % c_size != 0:
                    raise ValueError(
                        f"Field multiplicative order ({q - 1}) must be divisible by "
                        f"multiplicative coset size ({c_size}) for mixed code of length {q}"
                    )
                p_size = (q - 1) // c_size
            else:
                if length_int > q - 1 or length_int % c_size != 0:
                    raise ValueError(
                        f"Code length ({length_int}) must be <= {q - 1} and divisible by "
                        f"multiplicative coset size ({c_size}) for partition {i}"
                    )
                p_size = length_int // c_size

            self._partition_size.append(p_size)
            cosets = list_subgroup_cosets(base_field, self._sub_group[i], sgt_lit)[:p_size]
            self._partition.append(cosets)

        all_pts = {pt for part in self._partition for coset in part for pt in coset}
        self._evaluation_points: list[Any] = sort_field_elements(all_pts)
        if len(self._evaluation_points) != length_int:
            raise ValueError(
                f"Union of partition cosets has {len(self._evaluation_points)} evaluation points, "
                f"expected code length {length_int}"
            )

        pt_to_idx = {pt: idx for idx, pt in enumerate(self._evaluation_points)}
        self._evaluation_points_idx: list[list[list[int]]] = [
            [[pt_to_idx[h] for h in coset] for coset in part] for part in self._partition
        ]

        self._G, self._G_L, self._G_R = create_bipartite_coset_graph(
            self._partition, self._evaluation_points_idx, self._evaluation_points
        )

        self._registered_encoders = self._registered_encoders.copy()
        self._registered_decoders = self._registered_decoders.copy()
        super().__init__(base_field, length_int, "VectorEncoder", "IterativeDecoder")
        self._dimension = dim_int

        max_dim = self.max_dimension()
        if self._dimension < 1 or self._dimension > max_dim:
            raise ValueError(
                f"Code dimension ({self._dimension}) must satisfy 1 <= dimension <= {max_dim}"
            )

    def __eq__(self, other: object) -> bool:
        """Return True if ``other`` is a two-set Tamo-Barg code with identical parameters."""
        return (
            isinstance(other, TamoBargCodeTwoSets)
            and self.base_field() == other.base_field()
            and self.length() == other.length()
            and self.dimension() == other.dimension()
            and self.locality() == other.locality()
            and self.local_minimum_distance() == other.local_minimum_distance()
            and self.sub_group() == other.sub_group()
        )

    def __hash__(self) -> int:
        """Return a deterministic hash consistent with ``__eq__``."""
        sgs, sg_types = self.sub_group()
        return hash(
            (
                self.base_field(),
                self.length(),
                self.dimension(),
                tuple(self.locality()),
                tuple(self.local_minimum_distance()),
                tuple(tuple(sg) for sg in sgs),
                tuple(sg_types),
            )
        )

    def _repr_(self) -> str:
        """Return a human-readable summary of the code parameters."""
        return (
            f"[{self.length()}, {self.dimension()}, {self.locality()}] "
            f"Tamo-Barg Code over GF({self.base_field().cardinality()})"
        )

    def bipartite_graph(self) -> tuple[nx.MultiGraph, set[int], set[int]]:
        """Return ``(graph, left_nodes, right_nodes)`` for the two coset partitions."""
        return self._G, self._G_L, self._G_R

    def partition(self) -> tuple[list[list[list[Any]]], list[int]]:
        """Return ``(partitions, partition_sizes)`` across both recovery sets."""
        return self._partition, self._partition_size

    def sub_group(self) -> tuple[list[list[Any]], list[str]]:
        """Return ``(subgroups, subgroup_types)`` defining the two partitions."""
        return self._sub_group, self._sub_group_type

    def evaluation_points_idx(self) -> list[list[list[int]]]:
        """Return evaluation-point indices for each coset of each partition."""
        return self._evaluation_points_idx

    def evaluation_points(self) -> list[Any]:
        """Return the canonically sorted list of codeword evaluation points."""
        return self._evaluation_points

    def locality(self) -> list[int]:
        """Return the two-element list ``[r_0, r_1]`` of localities."""
        return self._locality

    def num_of_sets(self) -> int:
        """Return the number of disjoint-repair-set partitions (always ``2``)."""
        return self._num_of_sets

    def local_minimum_distance(self) -> list[int]:
        """Return the two-element list ``[d_0, d_1]`` of local minimum distances."""
        return self._local_minimum_distance

    def design_distance(self) -> int:
        """Return the polynomial degree-bound design distance ``n - max_deg``."""
        basis = self.encoder().enc_basis()
        max_degree = max((int(poly.degree()) for poly in basis), default=0)
        return int(self.length()) - max_degree

    def max_dimension(self) -> int:
        """Return the maximum message dimension supported by the intersection basis."""
        return int(self.encoder().max_dimension())

    def max_rate(self) -> Any:
        """Return the maximum achievable code rate ``max_dimension() / length()``."""
        return self.max_dimension() / self.length()

    def minimum_distance_py(self) -> int:
        """Compute the exact minimum Hamming weight by enumerating monic messages.

        Iterates only over projectively normalized non-zero message vectors in
        ``GF(q)^k`` whose first non-zero coordinate equals ``1``, achieving a
        ``(q - 1)x`` speedup over full message-space enumeration.

        Returns:
            Exact minimum Hamming distance of the code.
        """
        gen_mat = self.generator_matrix()
        field = self.base_field()
        k = int(self.dimension())
        min_wt = int(self.length())

        for lead in range(k):
            lead_row = gen_mat.row(lead)
            tail_dim = k - lead - 1
            if tail_dim == 0:
                wt = int(lead_row.hamming_weight())
                if 0 < wt < min_wt:
                    min_wt = wt
                    if min_wt == 1:
                        return 1
                continue
            sub_gen = gen_mat.matrix_from_rows(list(range(lead + 1, k)))
            for tail_vec in VectorSpace(field, tail_dim):
                cw = lead_row + tail_vec * sub_gen
                wt = int(cw.hamming_weight())
                if 0 < wt < min_wt:
                    min_wt = wt
                    if min_wt == 1:
                        return 1
        return min_wt


class TamoBargTwoSetsVectorEncoder(Encoder):
    """Polynomial intersection vector encoder for ``TamoBargCodeTwoSets``."""

    def __init__(self, code: TamoBargCodeTwoSets) -> None:
        """Construct per-partition Tamo-Barg bases and their degree-minimal intersection.

        Args:
            code: Associated ``TamoBargCodeTwoSets`` instance.

        Raises:
            ValueError: If ``code.dimension()`` is outside ``1 .. max_dimension``.
        """
        super().__init__(code)
        self._num_of_sets: int = code.num_of_sets()
        self._partition, self._partition_size = code.partition()
        self._sub_group, self._sub_group_type = code.sub_group()
        self._base_poly_ring = PolynomialRing(code.base_field(), "x")

        self._good_poly: list[Any] = []
        self._algebra_basis: list[list[Any]] = []
        self._enc_basis: list[list[Any]] = []

        for i in range(self._num_of_sets):
            sgt_lit: SubgroupType = "add" if self._sub_group_type[i] == "add" else "mult"
            good_p = build_good_polynomial(self._base_poly_ring, self._sub_group[i], sgt_lit)
            alg_b = build_algebra_basis(good_p, self._partition_size[i])
            enc_b = build_encoding_basis(self._base_poly_ring, alg_b, code.locality()[i])
            self._good_poly.append(good_p)
            self._algebra_basis.append(alg_b)
            self._enc_basis.append(enc_b)

        full_comb_basis = intersect_encoding_bases(
            self._base_poly_ring,
            self._enc_basis[0],
            self._enc_basis[1],
            int(code.length()),
        )
        self._max_dimension: int = len(full_comb_basis)
        k = int(code.dimension())
        if k < 1 or k > self._max_dimension:
            raise ValueError(
                f"Code dimension ({k}) must satisfy 1 <= dimension <= {self._max_dimension}"
            )
        self._comb_enc_basis: list[Any] = full_comb_basis[:k]

    def _repr_(self) -> str:
        """Return string representation of the vector encoder."""
        return f"Vector Encoder for {self.code()}"

    def enc_basis(self) -> list[Any]:
        """Return the ``k`` combined encoding basis polynomials of minimal degrees."""
        return self._comb_enc_basis

    def enc_basis_ext(self) -> list[list[Any]]:
        """Return the per-partition full encoding bases ``[basis_0, basis_1]``."""
        return self._enc_basis

    def enc_algebra_basis(self) -> list[list[Any]]:
        """Return the per-partition good-polynomial power bases."""
        return self._algebra_basis

    def enc_good_poly(self) -> list[Any]:
        """Return ``[g_0(x), g_1(x)]``, the good polynomials of both partitions."""
        return self._good_poly

    def enc_poly(self, message: Sequence[Any]) -> Any:
        """Evaluate ``sum_{i=0}^{k-1} message[i] * basis[i](x)`` in ``GF(q)['x']``.

        Args:
            message: Message vector or sequence of length ``k`` over ``GF(q)``.

        Returns:
            Univariate message polynomial in ``GF(q)['x']``.
        """
        return evaluate_message_polynomial(self._base_poly_ring, self._comb_enc_basis, message)

    def max_dimension(self) -> int:
        """Return the dimension of the full intersection encoding subspace."""
        return self._max_dimension

    @cached_method
    def generator_matrix(self) -> Any:
        """Build and cache the immutable ``k x n`` generator matrix over ``GF(q)``."""
        code = self.code()
        alphas = code.evaluation_points()
        gen_mat = matrix(
            code.base_field(),
            code.dimension(),
            code.length(),
            [[poly(alpha) for alpha in alphas] for poly in self._comb_enc_basis],
        )
        gen_mat.set_immutable()
        return gen_mat


class TamoBargIterativeDecoder(Decoder):
    """Iterative hard-decision local GRS decoder across both coset partitions."""

    def __init__(self, code: TamoBargCodeTwoSets, max_num_of_itr: int = 10) -> None:
        """Precompute local GRS codes, Gao decoders, and parity-check matrices.

        Args:
            code: Associated ``TamoBargCodeTwoSets`` instance.
            max_num_of_itr: Maximum number of alternating two-partition iterations.
        """
        if not _is_int_like(max_num_of_itr) or int(max_num_of_itr) < 0:
            raise ValueError(
                f"max_num_of_itr must be a non-negative integer, got {max_num_of_itr!r}"
            )
        super().__init__(code, code.ambient_space(), "VectorEncoder")
        self._max_num_of_itr: int = int(max_num_of_itr)
        self._F = code.base_field()
        self._partitions, _ = code.partition()
        self._evalpts_idx = code.evaluation_points_idx()

        self._partitions_local_codes: list[list[GeneralizedReedSolomonCode]] = []
        self._partitions_local_decoders: list[list[Any]] = []
        self._local_parity_checks: list[list[Any]] = []

        localities = code.locality()
        for i, partition in enumerate(self._partitions):
            code_tmp: list[GeneralizedReedSolomonCode] = []
            decoder_tmp: list[Any] = []
            pcm_tmp: list[Any] = []
            for coset in partition:
                local_grs = GeneralizedReedSolomonCode(coset, localities[i])
                code_tmp.append(local_grs)
                decoder_tmp.append(local_grs.decoder("Gao"))
                pcm_tmp.append(_safe_parity_check_matrix(local_grs))
            self._partitions_local_codes.append(code_tmp)
            self._partitions_local_decoders.append(decoder_tmp)
            self._local_parity_checks.append(pcm_tmp)

    def _repr_(self) -> str:
        """Return string representation of the iterative decoder."""
        return f"Iterative Decoder for {self.code()}"

    def _partition_is_syndrome_clean(self, part_idx: int, work: Any) -> bool:
        """Return True if every coset in partition ``part_idx`` has zero local syndrome."""
        for j, idxs in enumerate(self._evalpts_idx[part_idx]):
            local_vec = vector(self._F, [work[idx] for idx in idxs])
            if not (self._local_parity_checks[part_idx][j] * local_vec).is_zero():
                return False
        return True

    def decode_to_code(self, r: Any) -> tuple[Any, int]:
        """Iteratively decode cosets of both partitions without mutating ``r``.

        Args:
            r: Received word in ``GF(q)^n``.

        Returns:
            Tuple ``(corrected_word, num_itr)``.
        """
        work = copy(vector(self._F, r))
        num_itr = 0

        while num_itr < self._max_num_of_itr:
            uncorrectable_err = [False, False]
            changed_in_itr = False
            changed_in_p1 = False

            for i in (0, 1):
                for j, idxs in enumerate(self._evalpts_idx[i]):
                    local_vec = vector(self._F, [work[idx] for idx in idxs])
                    if (self._local_parity_checks[i][j] * local_vec).is_zero():
                        continue
                    try:
                        corr_c = self._partitions_local_decoders[i][j].decode_to_code(local_vec)
                    except (DecodingError, ValueError):
                        uncorrectable_err[i] = True
                        continue

                    if corr_c != local_vec:
                        changed_in_itr = True
                        if i == 1:
                            changed_in_p1 = True
                        for pos, idx in enumerate(idxs):
                            work[idx] = corr_c[pos]

            num_itr += 1
            if not uncorrectable_err[0] and not uncorrectable_err[1]:
                if not changed_in_p1 or self._partition_is_syndrome_clean(0, work):
                    break
            elif not changed_in_itr:
                break

        return work, num_itr

    def decode_to_message(self, r: Any) -> Any:
        """Decode ``r`` to its unencoded message vector."""
        corr_c, _ = self.decode_to_code(r)
        return self.code().unencode(corr_c)


class TamoBargIterativeEEDecoder(Decoder):
    """Iterative erasure-and-error local GRS decoder across both coset partitions."""

    def __init__(
        self,
        code: TamoBargCodeTwoSets,
        max_num_of_itr: int = 10,
        erasure_start_itr: int | None = None,
    ) -> None:
        """Precompute local GRS ErrorErasure decoders, syndromes, and polynomial ring.

        Args:
            code: Associated ``TamoBargCodeTwoSets`` instance.
            max_num_of_itr: Maximum number of alternating two-partition iterations.
            erasure_start_itr: Iteration index (1-based) at which DUE coset erasure
                marking begins; defaults to ``max_num_of_itr // 2``.
        """
        if not _is_int_like(max_num_of_itr) or int(max_num_of_itr) < 0:
            raise ValueError(
                f"max_num_of_itr must be a non-negative integer, got {max_num_of_itr!r}"
            )
        super().__init__(code, code.ambient_space(), "VectorEncoder")
        self._max_num_of_itr: int = int(max_num_of_itr)
        if erasure_start_itr is None:
            self._erasure_start_itr: int = self._max_num_of_itr // 2
        else:
            if not _is_int_like(erasure_start_itr) or int(erasure_start_itr) < 0:
                raise ValueError(
                    f"erasure_start_itr must be a non-negative integer, got {erasure_start_itr!r}"
                )
            self._erasure_start_itr = int(erasure_start_itr)

        self._F = code.base_field()
        self._poly_ring = PolynomialRing(self._F, "x")
        self._partitions, _ = code.partition()
        self._evalpts_idx = code.evaluation_points_idx()
        self._local_min_distances = code.local_minimum_distance()

        self._partitions_local_codes: list[list[GeneralizedReedSolomonCode]] = []
        self._partitions_local_decoders: list[list[Any]] = []
        self._local_parity_checks: list[list[Any]] = []

        localities = code.locality()
        for i, partition in enumerate(self._partitions):
            code_tmp: list[GeneralizedReedSolomonCode] = []
            decoder_tmp: list[Any] = []
            pcm_tmp: list[Any] = []
            for coset in partition:
                local_grs = GeneralizedReedSolomonCode(coset, localities[i])
                code_tmp.append(local_grs)
                decoder_tmp.append(local_grs.decoder("ErrorErasure"))
                pcm_tmp.append(_safe_parity_check_matrix(local_grs))
            self._partitions_local_codes.append(code_tmp)
            self._partitions_local_decoders.append(decoder_tmp)
            self._local_parity_checks.append(pcm_tmp)

    def _repr_(self) -> str:
        """Return string representation of the iterative erasure-and-error decoder."""
        return f"Iterative Decoder Error-and-Erasure for {self.code()}"

    def _partition_is_syndrome_clean(self, part_idx: int, work: Any) -> bool:
        """Return True if every coset in partition ``part_idx`` has zero local syndrome."""
        for j, idxs in enumerate(self._evalpts_idx[part_idx]):
            local_vec = vector(self._F, [work[idx] for idx in idxs])
            if not (self._local_parity_checks[part_idx][j] * local_vec).is_zero():
                return False
        return True

    def decode_to_code(self, r: Any) -> tuple[tuple[Any, Any], int]:
        """Iteratively decode errors and DUE-marked erasures without mutating ``r``.

        Args:
            r: Received word in ``GF(q)^n``.

        Returns:
            Tuple ``((corrected_word, erasure_vector), num_itr)``.
        """
        work = copy(vector(self._F, r))
        erasure_flags = [0] * len(work)
        num_itr = 0

        while num_itr < self._max_num_of_itr:
            prev_work = list(work)
            prev_erasures = list(erasure_flags)
            uncorrectable_err = [False, False]
            due_status: list[list[bool]] = []

            for i in (0, 1):
                d_local = self._local_min_distances[i]
                due_tmp: list[bool] = []
                for j, coset in enumerate(self._partitions[i]):
                    idxs = self._evalpts_idx[i][j]
                    r_list = [work[idx] for idx in idxs]
                    era_list = [erasure_flags[idx] for idx in idxs]
                    era_wt = sum(era_list)

                    if (
                        era_wt == 0
                        and (self._local_parity_checks[i][j] * vector(self._F, r_list)).is_zero()
                    ):
                        due_tmp.append(False)
                    elif era_wt >= d_local:
                        due_tmp.append(True)
                        uncorrectable_err[i] = True
                    elif era_wt == d_local - 1:
                        corr_list = interpolate_erasures(self._poly_ring, coset, r_list, era_list)
                        for pos, idx in enumerate(idxs):
                            work[idx] = corr_list[pos]
                        due_tmp.append(False)
                    else:
                        try:
                            corr_c = self._partitions_local_decoders[i][j].decode_to_code(
                                (vector(self._F, r_list), vector(GF(2), era_list))
                            )
                        except (DecodingError, ValueError):
                            due_tmp.append(True)
                            uncorrectable_err[i] = True
                        else:
                            for pos, idx in enumerate(idxs):
                                work[idx] = corr_c[pos]
                            due_tmp.append(False)

                due_status.append(due_tmp)

            num_itr += 1

            if (
                not uncorrectable_err[0]
                and not uncorrectable_err[1]
                and self._partition_is_syndrome_clean(0, work)
            ):
                for s in (0, 1):
                    for t, idxs in enumerate(self._evalpts_idx[s]):
                        if not due_status[s][t]:
                            for idx in idxs:
                                erasure_flags[idx] = 0
                break

            if num_itr >= self._erasure_start_itr:
                for t, idxs in enumerate(self._evalpts_idx[0]):
                    mark = 1 if due_status[0][t] else 0
                    for idx in idxs:
                        erasure_flags[idx] = mark
                for t, idxs in enumerate(self._evalpts_idx[1]):
                    if not due_status[1][t]:
                        for idx in idxs:
                            erasure_flags[idx] = 0

            if (
                num_itr > self._erasure_start_itr
                and list(work) == prev_work
                and erasure_flags == prev_erasures
            ):
                break

        return (work, vector(GF(2), erasure_flags)), num_itr

    def decode_to_message(self, r: Any) -> Any:
        """Decode ``r`` to its unencoded message vector."""
        (corr_c, _), _ = self.decode_to_code(r)
        return self.code().unencode(corr_c)


class TamoBargGlobalDecoder(Decoder):
    """Global GRS hard-decision decoder using the code's polynomial design distance."""

    def __init__(self, code: TamoBargCodeTwoSets) -> None:
        """Initialize the ambient global GRS code, Gao decoder, and parity-check matrix."""
        super().__init__(code, code.ambient_space(), "VectorEncoder")
        self._F = code.base_field()
        self._global_grs = GeneralizedReedSolomonCode(
            code.evaluation_points(),
            int(code.length()) - code.design_distance() + 1,
        )
        self._global_decoder = self._global_grs.decoder("Gao")
        self._global_pcm = _safe_parity_check_matrix(self._global_grs)

    def _repr_(self) -> str:
        """Return string representation of the global decoder."""
        return f"Global Decoder for {self.code()}"

    def decode_to_code(self, r: Any) -> Any:
        """Decode ``r`` via the global enveloping GRS code without mutating ``r``.

        Args:
            r: Received word in ``GF(q)^n``.

        Returns:
            Decoded codeword vector in ``GF(q)^n``.
        """
        work = copy(vector(self._F, r))
        if (self._global_pcm * work).is_zero():
            return work
        return self._global_decoder.decode_to_code(work)


class TamoBargGlobalEEDecoder(Decoder):
    """Global GRS erasure-and-error decoder over ``GF(q)^n x GF(2)^n``."""

    def __init__(self, code: TamoBargCodeTwoSets) -> None:
        """Initialize the Cartesian input space and global GRS ErrorErasure decoder."""
        input_space = cartesian_product(
            [code.ambient_space(), VectorSpace(GF(2), int(code.length()))]
        )
        super().__init__(code, input_space, "VectorEncoder")
        self._F = code.base_field()
        self._poly_ring = PolynomialRing(self._F, "x")
        self._global_grs = GeneralizedReedSolomonCode(
            code.evaluation_points(),
            int(code.length()) - code.design_distance() + 1,
        )
        self._global_decoder = self._global_grs.decoder("ErrorErasure")
        self._global_pcm = _safe_parity_check_matrix(self._global_grs)

    def _repr_(self) -> str:
        """Return string representation of the global erasure-and-error decoder."""
        return f"Global Erasure-and-Error Decoder for {self.code()}"

    def decode_to_code(self, word_and_erasure_vector: Any) -> Any:
        """Decode ``(word, erasure_vector)`` globally without mutating inputs.

        Args:
            word_and_erasure_vector: Pair ``(word, erasure_vector)`` in
                ``GF(q)^n x GF(2)^n``.

        Returns:
            Decoded codeword vector in ``GF(q)^n``.

        Raises:
            DecodingError: If ``erasure_vector`` exceeds ``d_design - 1`` erasures
                or error-and-erasure decoding fails.
        """
        word, erasure_vector = word_and_erasure_vector
        work = copy(vector(self._F, word))
        era_vec = copy(vector(GF(2), erasure_vector))
        era_wt = int(era_vec.hamming_weight())
        d_glob = int(self._global_grs.minimum_distance())

        if era_wt == 0 and (self._global_pcm * work).is_zero():
            return work
        if era_wt >= d_glob:
            raise DecodingError("Too many erasures for global decoder")
        if era_wt == d_glob - 1:
            corr_list = interpolate_erasures(
                self._poly_ring,
                self.code().evaluation_points(),
                list(work),
                list(era_vec),
            )
            return vector(self._F, corr_list)
        return self._global_decoder.decode_to_code((work, era_vec))


class TamoBargTwoStepDecoder(Decoder):
    """Two-step decoder combining iterative local decoding with global GRS cleanup."""

    def __init__(self, code: TamoBargCodeTwoSets, max_num_of_itr: int = 10) -> None:
        """Initialize the inner iterative decoder and outer global decoder."""
        super().__init__(code, code.ambient_space(), "VectorEncoder")
        self._iter_decoder = TamoBargIterativeDecoder(code, max_num_of_itr)
        self._global_decoder = TamoBargGlobalDecoder(code)

    def _repr_(self) -> str:
        """Return string representation of the two-step decoder."""
        return f"Two-Steps Decoder for {self.code()}"

    def decode_to_code(self, r: Any) -> tuple[Any, int]:
        """Run iterative local decoding followed by global GRS decoding.

        Args:
            r: Received word in ``GF(q)^n``.

        Returns:
            Tuple ``(corrected_word, num_itr)``.
        """
        tmp, num_itr = self._iter_decoder.decode_to_code(r)
        try:
            result = self._global_decoder.decode_to_code(tmp)
        except (DecodingError, ValueError):
            result = tmp
        return result, num_itr

    def decode_to_message(self, r: Any) -> Any:
        """Decode ``r`` to its unencoded message vector."""
        corr_c, _ = self.decode_to_code(r)
        return self.code().unencode(corr_c)


class TamoBargTwoStepEEDecoder(Decoder):
    """Two-step erasure-and-error decoder combining iterative EE with global EE cleanup."""

    def __init__(
        self,
        code: TamoBargCodeTwoSets,
        max_num_of_itr: int = 10,
        erasure_start_itr: int | None = None,
    ) -> None:
        """Initialize the inner iterative EE decoder and outer global EE decoder."""
        super().__init__(code, code.ambient_space(), "VectorEncoder")
        self._iter_ee_decoder = TamoBargIterativeEEDecoder(code, max_num_of_itr, erasure_start_itr)
        self._global_ee_decoder = TamoBargGlobalEEDecoder(code)

    def _repr_(self) -> str:
        """Return string representation of the two-step erasure-and-error decoder."""
        return f"Two-Steps Erasure-and-Error Decoder for {self.code()}"

    def decode_to_code(self, r: Any) -> tuple[Any, int, int]:
        """Run iterative EE decoding followed by global EE decoding.

        Args:
            r: Received word in ``GF(q)^n``.

        Returns:
            Tuple ``(corrected_word, num_itr, num_rem_era)``.
        """
        (word_after_iter, era_vec), num_itr = self._iter_ee_decoder.decode_to_code(r)
        num_rem_era = int(era_vec.hamming_weight())
        try:
            result = self._global_ee_decoder.decode_to_code((word_after_iter, era_vec))
        except (DecodingError, ValueError):
            result = word_after_iter
        return result, num_itr, num_rem_era

    def decode_to_message(self, r: Any) -> Any:
        """Decode ``r`` to its unencoded message vector."""
        corr_c, _, _ = self.decode_to_code(r)
        return self.code().unencode(corr_c)


TamoBargCodeTwoSets._registered_encoders["VectorEncoder"] = TamoBargTwoSetsVectorEncoder
TamoBargCodeTwoSets._registered_decoders["IterativeDecoder"] = TamoBargIterativeDecoder
TamoBargCodeTwoSets._registered_decoders["TwoStepsDecoder"] = TamoBargTwoStepDecoder
TamoBargCodeTwoSets._registered_decoders["GlobalDecoder"] = TamoBargGlobalDecoder
TamoBargCodeTwoSets._registered_decoders["IterativeErasureErrorDecoder"] = (
    TamoBargIterativeEEDecoder
)
TamoBargCodeTwoSets._registered_decoders["GlobalErasureErrorDecoder"] = TamoBargGlobalEEDecoder
TamoBargCodeTwoSets._registered_decoders["TwoStepsErasureErrorDecoder"] = TamoBargTwoStepEEDecoder
