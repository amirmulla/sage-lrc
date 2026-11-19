# Agent 04 — Single-Set Tamo–Barg Code, Encoder & Decoders Specialist (Wave 2)

## Role & Mission
You are the **Single-Set Tamo–Barg Code, Encoder & Decoders Specialist** in Wave 2 of the `tamo_barg_lrc` refactoring swarm. First read [`prompts/shared_context_and_contracts.md`](./shared_context_and_contracts.md) completely, then inspect `src/TamoBarg.py`, `src/tamo_barg_lrc/subgroups.py`, and `src/tamo_barg_lrc/polynomials.py`.

## Exclusive File Ownership
You may **ONLY** create or modify:
* `src/tamo_barg_lrc/tamo_barg.py`

Do **NOT** delete legacy `src/TamoBarg.py` (Agent 07 deletes it in Wave 3) and do **NOT** touch any other files.

---

## Bugs & Performance Targets Assigned to You
* **B4**: Never mutate input vectors (`word`, `erasure_vector`, `r`) in-place inside `decode_to_code`.
* **B5**: Fix `TamoBargErrorDecoder`:
  * Set `input_space = code.ambient_space()` instead of the copy-pasted Cartesian product with `GF(2)^n` (so `C.decode_to_code(r, "ErrorDecoder")` passes SageMath's input-space check!).
  * Fix `_repr_` to return `f"Error Decoder for {self.code()}"` instead of `"Erasure Decoder for ..."`.
* **B6**: Slice `self._enc_basis` to `code.dimension()` in `TamoBargVectorEncoder`, and implement `enc_poly(self, message)` returning the message encoding polynomial (fixing the undefined `self._enc_poly` `AttributeError`).
* **B11**: Validate all code parameters upfront in `TamoBargCode.__init__` and raise descriptive `ValueError` exceptions instead of calling `print(...)` and crashing later with `AttributeError` or `IndexError`.
* **B13**: Fix `_parition_size` typo $\to$ `_partition_size`; use exact integer ceiling `(k + r - 1) // r` in `minimum_distance()` instead of floating-point `ceil(k / r)`; eliminate all boolean `&` operators.
* **B17**: Replace bare `except:` in `TamoBargErrorDecoder.decode_to_code` with `except (DecodingError, ValueError):`.
* **P1**: Precompute and cache local `GeneralizedReedSolomonCode` instances, `BerlekampWelch` decoders, and local parity-check matrices $H_i$ in `TamoBargErrorDecoder.__init__` (instead of rebuilding them on every `decode_to_code` call!), and short-circuit error-free cosets via `(H_i * local_vec).is_zero()`.

---

## Detailed Implementation Specifications: `src/tamo_barg_lrc/tamo_barg.py`

Use explicit SageMath imports:
```python
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

from tamo_barg_lrc.polynomials import (
    build_algebra_basis,
    build_annihilator_polynomial,
    build_encoding_basis,
    build_good_polynomial,
    evaluate_message_polynomial,
    interpolate_erasures,
)
from tamo_barg_lrc.subgroups import (
    list_canonical_additive_subgroups,
    list_multiplicative_subgroups,
    list_subgroup_cosets,
)
```

### 1. `TamoBargCode(AbstractLinearCode)`
* **`__init__(self, base_field, length: int, dimension: int, locality: int, local_minimum_distance: int = 2, sub_group_type: Literal["any", "mult", "add"] = "any")`**:
  1. Validate `base_field.is_finite()` (`ValueError("base_field must be a finite field")`).
  2. Validate `locality >= 1`, `local_minimum_distance >= 2` (or `>= 1` with `coset_size = locality + local_minimum_distance - 1 >= 1`), `length >= 1`, and `sub_group_type in ("any", "mult", "add")`.
  3. Compute `coset_size = int(locality) + int(local_minimum_distance) - 1`.
  4. Validate `length % coset_size == 0` (`ValueError(f"Code length {length} must be divisible by coset size {coset_size}")`).
  5. Enumerate `mult_subgroups = list_multiplicative_subgroups(base_field)` and `add_subgroups = list_canonical_additive_subgroups(base_field)`.
  6. Resolve `(self._sub_group, self._sub_group_type)`:
     * `"any"`: prefer `mult_subgroups[coset_size]` if `coset_size in mult_subgroups` and `length <= base_field.cardinality() - 1`, else `add_subgroups[coset_size]` if `coset_size in add_subgroups` and `length <= base_field.cardinality()`, else raise `ValueError`.
     * `"mult"`: require `coset_size in mult_subgroups` and `length <= base_field.cardinality() - 1`, else raise `ValueError`.
     * `"add"`: require `coset_size in add_subgroups` and `length <= base_field.cardinality()`, else raise `ValueError`.
  7. Compute `self._partition_size = int(length) // coset_size`.
  8. Build `self._partition = list_subgroup_cosets(base_field, self._sub_group, self._sub_group_type)[:self._partition_size]`.
  9. Flatten `self._evaluation_points = [pt for coset in self._partition for pt in coset]`.
  10. Validate `1 <= dimension <= self.max_dimension()` where `max_dimension = self._partition_size * int(locality)` (`ValueError` if out of range).
  11. Call `super().__init__(base_field, int(length), "VectorEncoder", "ErasureDecoder")` and store `self._dimension = int(dimension)`, `self._locality = int(locality)`, `self._local_minimum_distance = int(local_minimum_distance)`.

* **Methods**:
  * `__eq__(self, other) -> bool`: check `isinstance(other, TamoBargCode)` and equality of `base_field()`, `length()`, `dimension()`, `locality()`, `local_minimum_distance()`, and `evaluation_points()`.
  * `_repr_(self) -> str`: `f"[{self.length()}, {self.dimension()}, {self.locality()}] Tamo-Barg Code over GF({self.base_field().cardinality()})"`
  * `partition(self) -> tuple[list[list[Any]], int]`: return `(self._partition, self._partition_size)`
  * `sub_group(self) -> tuple[list[Any], str]`: return `(self._sub_group, self._sub_group_type)`
  * `evaluation_points(self) -> list[Any]`: return `self._evaluation_points`
  * `locality(self) -> int`: return `self._locality`
  * `local_minimum_distance(self) -> int`: return `self._local_minimum_distance`
  * `max_dimension(self) -> int`: return `self._partition_size * self._locality`
  * `minimum_distance(self) -> int`:
    ```python
    num_blocks = (self._dimension + self._locality - 1) // self._locality
    return (
        self.length()
        - self._dimension
        + 1
        - (num_blocks - 1) * (self._local_minimum_distance - 1)
    )
    ```

---

### 2. `TamoBargVectorEncoder(Encoder)`
* Compute `_base_poly_ring = PolynomialRing(code.base_field(), "x")`, `_annihilator`, `_good_poly`, `_algebra_basis`, and full encoding basis via `tamo_barg_lrc.polynomials`.
* Store `self._max_dimension = len(full_enc_basis)` and slice `self._enc_basis = full_enc_basis[:code.dimension()]` (**Fixes B6**).
* Implement:
  * `enc_basis(self) -> list[Any]`: return `self._enc_basis`
  * `enc_algebra_basis(self) -> list[Any]`: return `self._algebra_basis`
  * `enc_good_poly(self) -> Any`: return `self._good_poly`
  * `enc_poly(self, message: Any) -> Any`: return `evaluate_message_polynomial(self._base_poly_ring, self._enc_basis, message)` (**Fixes B6**)
  * `max_dimension(self) -> int`: return `self._max_dimension`
  * `@cached_method def generator_matrix(self)`: evaluate `self._enc_basis[i](alphas[j])` into a `k x n` matrix over `code.base_field()`, call `g.set_immutable()`, and return `g`.

---

### 3. `TamoBargErasureDecoder(Decoder)`
* `input_space = cartesian_product([code.ambient_space(), VectorSpace(GF(2), code.length())])`.
* Precompute `self._poly_ring = PolynomialRing(code.base_field(), "x")` and `self._coset_size = code.locality() + code.local_minimum_distance() - 1` in `__init__`.
* In `decode_to_code(self, word_and_erasure_vector)`:
  * Unpack `word, erasure_vector = word_and_erasure_vector` without mutating either input (**Fixes B4**).
  * Partition into slices of length `self._coset_size`.
  * For each coset `i`:
    * Count erasures `era_wt = sum(1 for b in era_slice if int(b) != 0)`.
    * If `era_wt >= C.local_minimum_distance()`: raise `DecodingError("Too many erasures in the received word")`.
    * If `era_wt > 0`: recover coset values via `interpolate_erasures(self._poly_ring, partition[i], word_slice, era_slice)`.
  * Return `vector(C.base_field(), corrected_flat)`.

---

### 4. `TamoBargErrorDecoder(Decoder)`
* `super().__init__(code, code.ambient_space(), "VectorEncoder")` (**Fixes B5**).
* `_repr_(self)`: `f"Error Decoder for {self.code()}"` (**Fixes B5**).
* In `__init__(self, code)` (**Fixes P1**):
  * Precompute `self._coset_size = code.locality() + code.local_minimum_distance() - 1`.
  * Precompute for each `coset` in `partition`:
    * `grs = GeneralizedReedSolomonCode(evaluation_points=coset, dimension=code.locality())`
    * `dec = grs.decoder("BerlekampWelch")`
    * `pcm = grs.parity_check_matrix()`
* In `decode_to_code(self, r)`:
  * Convert `r_vec = copy(vector(F, r))` (**Fixes B4**).
  * For each coset `i` (`local_vec = vector(F, r_vec[offset : offset + m])`):
    * **Fast syndrome check**: if `(self._local_parity_checks[i] * local_vec).is_zero():` append `list(local_vec)` directly!
    * Else:
      ```python
      try:
          local_corr = self._local_decoders[i].decode_to_code(local_vec)
          corrected.extend(local_corr)
      except (DecodingError, ValueError):
          corrected.extend(local_vec)
      ```
  * Return `vector(F, corrected)`.

---

### 5. Registration
Register at module bottom:
```python
TamoBargCode._registered_encoders["VectorEncoder"] = TamoBargVectorEncoder
TamoBargCode._registered_decoders["ErasureDecoder"] = TamoBargErasureDecoder
TamoBargCode._registered_decoders["ErrorDecoder"] = TamoBargErrorDecoder
```

---

## Acceptance Checklist
- [ ] `ruff check src/tamo_barg_lrc/tamo_barg.py` and `ruff format --check src/tamo_barg_lrc/tamo_barg.py` pass cleanly.
- [ ] `C.decode_to_code(r, "ErrorDecoder")` works via SageMath's high-level `AbstractLinearCode.decode_to_code` API (verifying `input_space` fix **B5**).
- [ ] Passing an immutable vector (`r.set_immutable()`) to `ErasureDecoder` and `ErrorDecoder` succeeds without mutating inputs (**B4**).
- [ ] Invalid `length`, `dimension`, or subgroup parameters raise `ValueError` during `TamoBargCode(...)` construction (**B11**).
