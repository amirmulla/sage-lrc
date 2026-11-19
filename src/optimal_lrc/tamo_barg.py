"""Single-set Tamo-Barg optimal locally recoverable codes, encoders, and decoders."""

from copy import copy
from typing import Any, Literal

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

from optimal_lrc.polynomials import (
    build_algebra_basis,
    build_annihilator_polynomial,
    build_encoding_basis,
    build_good_polynomial,
    evaluate_message_polynomial,
    interpolate_erasures,
)
from optimal_lrc.subgroups import (
    SubgroupType,
    list_canonical_additive_subgroups,
    list_multiplicative_subgroups,
    list_subgroup_cosets,
)

__all__ = [
    "TamoBargCode",
    "TamoBargVectorEncoder",
    "TamoBargErasureDecoder",
    "TamoBargErrorDecoder",
]


def _coerce_int(value: Any, name: str) -> int:
    """Validate and convert an integer-valued code parameter to ``int``."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    try:
        int_val = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc
    if int_val != value:
        raise ValueError(f"{name} must be an integer, got {value!r}")
    return int_val


class TamoBargCode(AbstractLinearCode):
    """Single-set Tamo-Barg optimal locally recoverable code over ``GF(q)``.

    Constructs an ``(r, d_local)``-locally recoverable evaluation code whose
    evaluation points are partitioned into disjoint cosets of size
    ``r + d_local - 1`` of a multiplicative or additive subgroup of ``GF(q)``.
    """

    _registered_encoders: dict[str, type[Encoder]] = {}
    _registered_decoders: dict[str, type[Decoder]] = {}

    def __init__(
        self,
        base_field: Any,
        length: int,
        dimension: int,
        locality: int,
        local_minimum_distance: int = 2,
        sub_group_type: Literal["any", "mult", "add"] = "any",
    ) -> None:
        """Initialize a single-set Tamo-Barg code.

        Args:
            base_field: Finite field ``GF(q)``.
            length: Codeword block length ``n`` (divisible by ``r + d_local - 1``).
            dimension: Message dimension ``k`` (``1 <= k <= max_dimension``).
            locality: Locality parameter ``r >= 1``.
            local_minimum_distance: Local code minimum distance ``d_local >= 2``.
            sub_group_type: Subgroup family selector: ``"any"``, ``"mult"``, or ``"add"``.

        Raises:
            ValueError: If any parameter is invalid or no suitable subgroup exists.
        """
        if (
            not hasattr(base_field, "is_finite")
            or not base_field.is_finite()
            or (hasattr(base_field, "is_field") and not base_field.is_field())
        ):
            raise ValueError("base_field must be a finite field")

        n_int = _coerce_int(length, "length")
        k_int = _coerce_int(dimension, "dimension")
        r_int = _coerce_int(locality, "locality")
        d_local_int = _coerce_int(local_minimum_distance, "local_minimum_distance")

        if n_int < 1:
            raise ValueError(f"length ({n_int}) must be at least 1")
        if r_int < 1:
            raise ValueError(f"locality ({r_int}) must be at least 1")
        if d_local_int < 2:
            raise ValueError(f"local_minimum_distance ({d_local_int}) must be at least 2")
        if sub_group_type not in ("any", "mult", "add"):
            raise ValueError(
                f"sub_group_type must be 'any', 'mult', or 'add', got {sub_group_type!r}"
            )

        coset_size = r_int + d_local_int - 1
        if n_int % coset_size != 0:
            raise ValueError(f"Code length {n_int} must be divisible by coset size {coset_size}")

        q = int(base_field.cardinality())
        mult_subgroups = list_multiplicative_subgroups(base_field)
        add_subgroups = list_canonical_additive_subgroups(base_field)

        if sub_group_type == "any":
            if coset_size in mult_subgroups and n_int <= q - 1:
                self._sub_group = mult_subgroups[coset_size]
                self._sub_group_type: SubgroupType = "mult"
            elif coset_size in add_subgroups and n_int <= q:
                self._sub_group = add_subgroups[coset_size]
                self._sub_group_type = "add"
            else:
                raise ValueError(
                    f"No multiplicative or additive subgroup of size {coset_size} in "
                    f"{base_field} can support code length {n_int}"
                )
        elif sub_group_type == "mult":
            if coset_size not in mult_subgroups:
                raise ValueError(
                    f"No multiplicative subgroup of size {coset_size} exists in {base_field}"
                )
            if n_int > q - 1:
                raise ValueError(
                    f"Code length {n_int} exceeds multiplicative group order {q - 1} "
                    f"of {base_field}"
                )
            self._sub_group = mult_subgroups[coset_size]
            self._sub_group_type = "mult"
        else:
            if coset_size not in add_subgroups:
                raise ValueError(
                    f"No canonical additive subgroup of size {coset_size} exists in {base_field}"
                )
            if n_int > q:
                raise ValueError(
                    f"Code length {n_int} exceeds field cardinality {q} of {base_field}"
                )
            self._sub_group = add_subgroups[coset_size]
            self._sub_group_type = "add"

        self._partition_size = n_int // coset_size
        all_cosets = list_subgroup_cosets(base_field, self._sub_group, self._sub_group_type)
        if len(all_cosets) < self._partition_size:
            raise ValueError(
                f"Only {len(all_cosets)} cosets available, need {self._partition_size}"
            )
        self._partition = all_cosets[: self._partition_size]
        self._evaluation_points = [pt for coset in self._partition for pt in coset]

        self._locality = r_int
        self._local_minimum_distance = d_local_int
        max_dim = self._partition_size * self._locality
        if not (1 <= k_int <= max_dim):
            raise ValueError(f"Code dimension {k_int} must satisfy 1 <= dimension <= {max_dim}")

        super().__init__(base_field, n_int, "VectorEncoder", "ErasureDecoder")
        self._dimension = k_int

    def __eq__(self, other: object) -> bool:
        """Check structural equality between two single-set Tamo-Barg codes."""
        return (
            isinstance(other, TamoBargCode)
            and self.base_field() == other.base_field()
            and self.length() == other.length()
            and self.dimension() == other.dimension()
            and self.locality() == other.locality()
            and self.local_minimum_distance() == other.local_minimum_distance()
            and self.evaluation_points() == other.evaluation_points()
        )

    def __hash__(self) -> int:
        """Return a hash consistent with ``__eq__``."""
        return hash(
            (
                self.base_field(),
                int(self.length()),
                self._dimension,
                self._locality,
                self._local_minimum_distance,
                tuple(self._evaluation_points),
            )
        )

    def _repr_(self) -> str:
        """Return SageMath string representation of the code."""
        return (
            f"[{self.length()}, {self.dimension()}, {self.locality()}] "
            f"Tamo-Barg Code over GF({self.base_field().cardinality()})"
        )

    def partition(self) -> tuple[list[list[Any]], int]:
        """Return ``(coset_partition, partition_size)``."""
        return self._partition, self._partition_size

    def sub_group(self) -> tuple[list[Any], str]:
        """Return ``(subgroup_elements, sub_group_type)``."""
        return self._sub_group, self._sub_group_type

    def evaluation_points(self) -> list[Any]:
        """Return flattened list of ``n`` distinct evaluation points in ``GF(q)``."""
        return self._evaluation_points

    def locality(self) -> int:
        """Return the locality parameter ``r``."""
        return self._locality

    def local_minimum_distance(self) -> int:
        """Return the local repair group minimum distance ``d_local``."""
        return self._local_minimum_distance

    def max_dimension(self) -> int:
        """Return the maximum supported dimension ``partition_size * locality``."""
        return self._partition_size * self._locality

    def minimum_distance(self) -> int:
        """Return the Singleton-like optimal LRC minimum distance of the code."""
        num_blocks = (self._dimension + self._locality - 1) // self._locality
        return (
            int(self.length())
            - self._dimension
            + 1
            - (num_blocks - 1) * (self._local_minimum_distance - 1)
        )


class TamoBargVectorEncoder(Encoder):
    """Polynomial evaluation vector encoder for :class:`TamoBargCode`."""

    def __init__(self, code: TamoBargCode) -> None:
        """Initialize the encoder and construct the ``k``-element encoding polynomial basis.

        Args:
            code: Associated :class:`TamoBargCode` instance.

        Raises:
            ValueError: If ``code.dimension()`` exceeds the maximum basis size.
        """
        super().__init__(code)
        self._partition, self._partition_size = code.partition()
        sub_group, sub_group_type = code.sub_group()
        self._sub_group = sub_group
        self._sub_group_type = sub_group_type
        self._base_poly_ring = PolynomialRing(code.base_field(), "x")
        self._annihilator = build_annihilator_polynomial(
            self._base_poly_ring, code.evaluation_points()
        )
        if sub_group_type not in ("add", "mult"):
            raise ValueError(f"sub_group_type must be 'add' or 'mult', got {sub_group_type!r}")
        self._good_poly = build_good_polynomial(
            self._base_poly_ring,
            self._sub_group,
            sub_group_type,
            annihilator=self._annihilator,
        )
        self._algebra_basis = build_algebra_basis(
            self._good_poly,
            self._partition_size,
            annihilator=self._annihilator,
        )
        full_enc_basis = build_encoding_basis(
            self._base_poly_ring,
            self._algebra_basis,
            code.locality(),
            annihilator=self._annihilator,
        )
        self._max_dimension = len(full_enc_basis)
        if code.dimension() > self._max_dimension:
            raise ValueError(
                f"Code dimension {code.dimension()} exceeds maximum dimension {self._max_dimension}"
            )
        self._enc_basis = full_enc_basis[: code.dimension()]

    def __eq__(self, other: object) -> bool:
        """Check equality between two :class:`TamoBargVectorEncoder` instances."""
        return isinstance(other, TamoBargVectorEncoder) and self.code() == other.code()

    def __hash__(self) -> int:
        """Return a hash consistent with ``__eq__``."""
        return hash((type(self), self.code()))

    def _repr_(self) -> str:
        """Return SageMath string representation of the encoder."""
        return f"Vector Encoder for {self.code()}"

    def enc_basis(self) -> list[Any]:
        """Return the ``k`` encoding basis polynomials of minimum ascending degrees."""
        return self._enc_basis

    def enc_algebra_basis(self) -> list[Any]:
        """Return powers ``[1, g, ..., g^(partition_size - 1)]`` of the good polynomial."""
        return self._algebra_basis

    def enc_good_poly(self) -> Any:
        """Return the good polynomial ``g(x)`` constant on each subgroup coset."""
        return self._good_poly

    def enc_poly(self, message: Any) -> Any:
        """Construct the message encoding polynomial ``sum_{i=0}^{k-1} m_i * b_i(x)``.

        Args:
            message: Message vector or sequence of length ``k`` over ``GF(q)``.

        Returns:
            Univariate encoding polynomial in ``GF(q)['x']``.
        """
        return evaluate_message_polynomial(self._base_poly_ring, self._enc_basis, message)

    def max_dimension(self) -> int:
        """Return the total size of the unsliced encoding basis."""
        return self._max_dimension

    @cached_method
    def generator_matrix(self) -> Any:
        """Return the immutable ``k x n`` generator matrix over ``GF(q)``."""
        code = self.code()
        alphas = code.evaluation_points()
        g = matrix(
            code.base_field(),
            code.dimension(),
            code.length(),
            [[poly(alpha) for alpha in alphas] for poly in self._enc_basis],
        )
        g.set_immutable()
        return g


class TamoBargErasureDecoder(Decoder):
    """Local coset Lagrange-interpolation erasure decoder for :class:`TamoBargCode`."""

    def __init__(self, code: TamoBargCode) -> None:
        """Initialize the local erasure decoder.

        Args:
            code: Associated :class:`TamoBargCode` instance.
        """
        input_space = cartesian_product([code.ambient_space(), VectorSpace(GF(2), code.length())])
        super().__init__(code, input_space, "VectorEncoder")
        self._poly_ring = PolynomialRing(code.base_field(), "x")
        self._coset_size = code.locality() + code.local_minimum_distance() - 1

    def __eq__(self, other: object) -> bool:
        """Check equality between two :class:`TamoBargErasureDecoder` instances."""
        return isinstance(other, TamoBargErasureDecoder) and self.code() == other.code()

    def __hash__(self) -> int:
        """Return a hash consistent with ``__eq__``."""
        return hash((type(self), self.code()))

    def _repr_(self) -> str:
        """Return SageMath string representation of the erasure decoder."""
        return f"Erasure Decoder for {self.code()}"

    def decode_to_code(self, word_and_erasure_vector: Any) -> Any:
        """Recover up to ``d_local - 1`` erasures per repair coset without mutating inputs.

        Args:
            word_and_erasure_vector: Pair ``(word, erasure_vector)`` in
                ``GF(q)^n x GF(2)^n``.

        Returns:
            Corrected codeword vector in ``GF(q)^n``.

        Raises:
            DecodingError: If any coset contains ``>= d_local`` erasures.
            ValueError: If ``word`` or ``erasure_vector`` does not have length ``n``.
        """
        code = self.code()
        base_field = code.base_field()
        n = code.length()
        word, erasure_vector = word_and_erasure_vector
        if len(word) != n or len(erasure_vector) != n:
            raise ValueError(
                f"word and erasure_vector must have length {n}, "
                f"got {len(word)} and {len(erasure_vector)}"
            )

        m = self._coset_size
        d_local = code.local_minimum_distance()
        partition, partition_size = code.partition()
        corrected_flat: list[Any] = []

        for i in range(partition_size):
            offset = i * m
            word_slice = [base_field(word[j]) for j in range(offset, offset + m)]
            era_slice = [int(erasure_vector[j]) for j in range(offset, offset + m)]
            era_wt = sum(1 for b in era_slice if b != 0)
            if era_wt >= d_local:
                raise DecodingError("Too many erasures in the received word")
            if era_wt > 0:
                word_slice = interpolate_erasures(
                    self._poly_ring, partition[i], word_slice, era_slice
                )
            corrected_flat.extend(word_slice)

        return vector(base_field, corrected_flat)


class TamoBargErrorDecoder(Decoder):
    """Local GRS Berlekamp-Welch error decoder for :class:`TamoBargCode`."""

    def __init__(self, code: TamoBargCode) -> None:
        """Initialize the local error decoder and precompute coset GRS decoders and PCMs.

        Args:
            code: Associated :class:`TamoBargCode` instance.
        """
        super().__init__(code, code.ambient_space(), "VectorEncoder")
        self._coset_size = code.locality() + code.local_minimum_distance() - 1
        partition, _ = code.partition()
        self._local_codes: list[Any] = []
        self._local_decoders: list[Any] = []
        self._local_parity_checks: list[Any] = []
        for coset in partition:
            grs = GeneralizedReedSolomonCode(evaluation_points=coset, dimension=code.locality())
            self._local_codes.append(grs)
            self._local_decoders.append(grs.decoder("BerlekampWelch"))
            self._local_parity_checks.append(grs.parity_check_matrix())

    def __eq__(self, other: object) -> bool:
        """Check equality between two :class:`TamoBargErrorDecoder` instances."""
        return isinstance(other, TamoBargErrorDecoder) and self.code() == other.code()

    def __hash__(self) -> int:
        """Return a hash consistent with ``__eq__``."""
        return hash((type(self), self.code()))

    def _repr_(self) -> str:
        """Return SageMath string representation of the error decoder."""
        return f"Error Decoder for {self.code()}"

    def decode_to_code(self, r: Any) -> Any:
        """Locally decode each coset via syndrome check + Berlekamp-Welch without mutation.

        Args:
            r: Received vector in ``GF(q)^n``.

        Returns:
            Locally decoded vector in ``GF(q)^n``.

        Raises:
            ValueError: If ``r`` does not have length ``n``.
        """
        code = self.code()
        base_field = code.base_field()
        n = code.length()
        if len(r) != n:
            raise ValueError(f"Received word must have length {n}, got {len(r)}")

        r_vec = copy(vector(base_field, r))
        m = self._coset_size
        corrected: list[Any] = []

        for i in range(len(self._local_decoders)):
            offset = i * m
            local_vec = vector(base_field, r_vec[offset : offset + m])
            if (self._local_parity_checks[i] * local_vec).is_zero():
                corrected.extend(local_vec)
            else:
                try:
                    local_corr = self._local_decoders[i].decode_to_code(local_vec)
                    corrected.extend(local_corr)
                except (DecodingError, ValueError):
                    corrected.extend(local_vec)

        return vector(base_field, corrected)


TamoBargCode._registered_encoders["VectorEncoder"] = TamoBargVectorEncoder
TamoBargCode._registered_decoders["ErasureDecoder"] = TamoBargErasureDecoder
TamoBargCode._registered_decoders["ErrorDecoder"] = TamoBargErrorDecoder
