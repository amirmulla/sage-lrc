# Agent 02 — Finite-Field Subgroups & Polynomial Algebra Specialist (Wave 1)

## Role & Mission
You are the **Finite-Field Subgroups & Polynomial Algebra Specialist** in Wave 1 of the `optimal_lrc` refactoring swarm. First read [`prompts/shared_context_and_contracts.md`](./shared_context_and_contracts.md) completely, then inspect the legacy implementations in `src/AdditiveSubgroup.py`, `src/TamoBarg.py`, and `src/TamoBargTwoSets.py` for reference.

## Exclusive File Ownership
You may **ONLY** create or modify:
* `src/optimal_lrc/subgroups.py`
* `src/optimal_lrc/polynomials.py`

Do **NOT** delete legacy `src/*.py` files (Agent 07 handles legacy cleanup in Wave 3) and do **NOT** touch any other files.

---

## Bugs & Performance Targets Assigned to You
* **B7**: Fix `AdditiveSubgroup.py` string-concatenation base-$p$ bug (`''.join(str(i) for i in vec)` + `int(binary_string, p)`), which corrupts multi-digit residues for $p \ge 10$, crashes with `ValueError` for $p > 36$, and fails symbolically when `size` is not a power of $p$.
* **B8**: Fix `_list_add_subgroups` omitting $i = 0$ ($\{0\}$) and $i = \deg(F)$ ($GF(q)$), and fix `_list_sub_group_cosets` missing the $\{0\}$ coset when $H = \{0\}$ and returning non-deterministic `set(...)` orderings.
* **B13**: Eliminate bitwise `&` boolean precedence bugs (`j**i == 1 & first_identity_found == 1`).
* **P3**: Replace $O(q^2)$ multiplicative subgroup enumeration with $O(q)$ primitive-element divisor generation; replace $O(q|H|)$ duplicate coset work with $O(q)$ `visited`-set partitioning; replace OOM `list(combinations(elements, n))` in brute-force subgroup search with a lazy iterator + $O(1)$ `set` closure check.
* **P5**: Eliminate `PolynomialQuotientRing` wrapper overhead in favor of direct FLINT univariate `PolynomialRing(F, 'x')` arithmetic, product-tree annihilator construction, $O(1)$ `poly_ring(list(row))` conversion, and reverse-column RREF for degree-minimizing two-set basis intersection.

---

## Detailed Specifications: `src/optimal_lrc/subgroups.py`

Use explicit SageMath imports (`from sage.arith.misc import divisors`, `from sage.modules.free_module import VectorSpace`, `from sage.rings.finite_rings.finite_field_constructor import GF`, etc.)—**never** `from sage.all import *`.

Implement the exact public API required by `shared_context_and_contracts.md`:

1. **`field_element_to_int(elem: Any) -> int` & `sort_field_elements(elements: Iterable[Any]) -> list[Any]`**:
   * Map any finite field element (whether prime field `GF(p)` or extension field `GF(p^m)` with `'int'`, `'poly'`, or Givaro/NTL/PARI representation) deterministically to `0 .. q - 1`:
     ```python
     def field_element_to_int(elem: Any) -> int:
         if hasattr(elem, "to_integer"):
             return int(elem.to_integer())
         return int(elem)
     ```
   * Note: Ensure `elem == 0` (i.e. `elem.is_zero()`) always maps to `0`.
   * `sort_field_elements(elements)` sorts using `key=field_element_to_int`.

2. **`is_closed_under_addition(subgroup: Iterable[Any]) -> bool`**:
   * Materialize `elem_set = set(subgroup)`. Return `False` if `not elem_set`.
   * Check additive identity (`next(iter(elem_set)).parent().zero() in elem_set`) as an $O(1)$ fast fail.
   * Check `(a + b) in elem_set` for all pairs `(a, b)` (using unordered pairs `i <= j` since finite field addition is commutative, cutting checks in half!).

3. **`find_subspaces(dim: int, ambient_dim: int, base_field: Any) -> list[Any]`**:
   * Validate `0 <= dim <= ambient_dim` (raise `ValueError` otherwise).
   * Construct `prime_field = GF(base_field.characteristic())` and `V = VectorSpace(prime_field, ambient_dim)`.
   * Return `list(V.subspaces(dim))`.

4. **`find_additive_subgroups(field: Any, size: int) -> list[list[Any]]`** (**Fixes B7**):
   * Validate `field.is_finite()` and `size >= 1`.
   * Determine exact exponent `dim` such that `p ** dim == size` (where `p = int(field.characteristic())`, `m = int(field.degree())`):
     * Compute integer log via repeated division by `p`; if `size` is not an exact power of `p` or `dim > m`, raise `ValueError(f"Additive subgroup size {size} must be a power of characteristic {p} dividing {field.cardinality()}")`.
   * Precompute field $\mathbb{F}_p$-basis `field_basis = [alpha ** j for j in range(m)]` where `alpha = field.gen()`.
   * For each subspace $W$ in `find_subspaces(dim, m, field)`:
     * Convert each vector `vec` in `W` via `sum((field(int(coord)) * field_basis[j] for j, coord in enumerate(vec)), field.zero())`—**zero string concatenation or `int(..., base)` calls**, supporting arbitrary prime characteristics ($p = 2, 3, 13, 37, 101, \dots$)!
     * Sort each subgroup with `sort_field_elements`.
   * Sort the list of subgroups deterministically by `[field_element_to_int(x) for x in sg]` and return.

5. **`find_additive_subgroups_bruteforce(field: Any, size: int, max_subgroups: int = 1000) -> list[list[Any]]`** (**Fixes P3**):
   * Validate `size >= 1` and `field.cardinality() % size == 0`.
   * Since every additive subgroup contains `0`, fix `zero = field.zero()` and iterate lazily over `combinations(non_zero_elements, size - 1)` (never materializing `list(combinations(...))` in memory!).
   * Check `is_closed_under_addition([zero, *combo])`; sort with `sort_field_elements` and stop once `len(subgroups) >= max_subgroups`.

6. **`list_multiplicative_subgroups(field: Any) -> dict[int, list[Any]]`** (**Fixes B13 & P3**):
   * Validate `field.is_finite()`.
   * Let `q_minus_1 = int(field.cardinality()) - 1` and `g = field.multiplicative_generator()`.
   * For each `d` in `divisors(q_minus_1)`:
     * `d_int = int(d)`
     * `step = g ** (q_minus_1 // d_int)`
     * Build `[1, step, step^2, ..., step^(d_int - 1)]` via incremental multiplication `curr *= step` (no exponentiation inside inner loop!).
     * Store `subgroups[d_int] = sort_field_elements(elems)`.
   * Return `subgroups`.

7. **`list_canonical_additive_subgroups(field: Any) -> dict[int, list[Any]]`** (**Fixes B8**):
   * Let `p = int(field.characteristic())`, `m = int(field.degree())`, `alpha = field.gen()`.
   * Start with `current = [field.zero()]` and `result = {1: list(current)}` ($i = 0$).
   * For `i` in `range(m)` (building sizes $p^1, p^2, \dots, p^m$ iteratively in $O(q)$ total additions!):
     * `basis_elem = alpha ** i`
     * `current = [h + c * basis_elem for c in range(p) for h in current]`
     * `result[len(current)] = sort_field_elements(current)`
   * Notice how clean and fast this is: builds **all** $m + 1$ canonical additive subgroups (including size $1$ and size $q$) in $O(q)$ total field operations without creating temporary `GF(p**i)` fields or string conversions!

8. **`shift_additive_subgroup(field: Any, subgroup: Sequence[Any], shift_exponent: int) -> list[Any]`**:
   * Multiply every element of `subgroup` by `field.multiplicative_generator() ** int(shift_exponent)` (note: `field.primitive_element()` or `field.multiplicative_generator()`—verify they match in SageMath `GF(q)`: for extension fields `field.multiplicative_generator()` is always defined on prime and extension fields, whereas `primitive_element()` is only on extension fields! Wait: check whether legacy `TamoBargTwoSets.py` used `base_field.primitive_element()`! On `GF(p^m)` in SageMath, `F.multiplicative_generator()` returns a generator of $F^\times$—specifically `a` when the Conway/primitive modulus makes `a` primitive—check `F.primitive_element()` fallback: `gen = field.primitive_element() if hasattr(field, "primitive_element") and field.degree() > 1 else field.multiplicative_generator()` so exact subgroup compatibility with legacy `GF(64, 'a', repr='int')` tests is preserved!).
   * Return `sort_field_elements(shifted)`.

9. **`list_subgroup_cosets(field: Any, subgroup: Sequence[Any], subgroup_type: SubgroupType) -> list[list[Any]]`** (**Fixes B8 & P3**):
   * Validate `subgroup_type in ("add", "mult")` and `len(subgroup) >= 1`.
   * Ordered ambient candidates:
     * `"add"`: `sort_field_elements(field)` (includes `0`!)
     * `"mult"`: `[x for x in sort_field_elements(field) if not x.is_zero()]`
   * Maintain `visited: set[Any] = set()` and `cosets: list[list[Any]] = []`:
     * For `g` in candidates:
       * If `g in visited`: `continue`
       * Compute `coset = sort_field_elements(h + g if subgroup_type == "add" else h * g for h in subgroup)`
       * `visited.update(coset)`
       * `cosets.append(coset)`
   * Return `cosets` (deterministic order, $O(q)$ field ops!).

---

## Detailed Specifications: `src/optimal_lrc/polynomials.py`

Implement fast FLINT-backed univariate polynomial utilities (**Fixes P5**):

1. **`build_annihilator_polynomial(poly_ring: Any, evaluation_points: Sequence[Any]) -> Any`**:
   * `x = poly_ring.gen()`
   * Return `prod((x - pt for pt in evaluation_points), poly_ring.one())`.

2. **`build_good_polynomial(poly_ring: Any, subgroup: Sequence[Any], subgroup_type: Literal["add", "mult"], annihilator: Any | None = None) -> Any`**:
   * Validate `subgroup_type in ("add", "mult")`.
   * `x = poly_ring.gen()`
   * If `subgroup_type == "mult"`: `good = x ** len(subgroup)`
   * Else: `good = prod((x - h for h in subgroup), poly_ring.one())`
   * If `annihilator is not None and good.degree() >= annihilator.degree()`: `good = good % annihilator`
   * Return `good`.

3. **`build_algebra_basis(good_poly: Any, partition_size: int, annihilator: Any | None = None) -> list[Any]`**:
   * Incrementally compute `[1, g, g^2, ..., g^(partition_size - 1)]`, reducing `% annihilator` only if `annihilator is not None and curr.degree() >= annihilator.degree()`.

4. **`build_encoding_basis(poly_ring: Any, algebra_basis: Sequence[Any], locality: int, annihilator: Any | None = None) -> list[Any]`**:
   * Precompute `[x ** i for i in range(locality)]`.
   * Form products `p * x_pow` (reducing `% annihilator` if needed).
   * Sort deterministically by `(poly.degree(), [field_element_to_int(c) for c in reversed(poly.list())])`.

5. **`intersect_encoding_bases(poly_ring: Any, basis_a: Sequence[Any], basis_b: Sequence[Any], length: int) -> list[Any]`**:
   * Compute the intersection $\text{span}(B_a) \cap \text{span}(B_b) \subseteq F[x]_{< \text{length}}$ and return its **degree-minimizing basis** sorted in strictly ascending degree:
     1. Handle empty input bases cleanly (return `[]`).
     2. Build `F = poly_ring.base_ring()`, `W_a = matrix(F, len(basis_a), length, [p.padded_list(length) for p in basis_a])`, and `W_b = matrix(F, len(basis_b), length, [p.padded_list(length) for p in basis_b])`.
     3. Intersect row spaces via `inter_space = W_a.row_space().intersection(W_b.row_space())` (or Zassenhaus kernel of `W_a.stack(-W_b)`). If `inter_space.dimension() == 0`, return `[]`.
     4. Let `B_int = inter_space.basis_matrix()` ($k_{\max} \times \text{length}$).
     5. **Reverse-column RREF** so pivots eliminate highest degrees first:
        ```python
        rev_cols = list(range(length - 1, -1, -1))
        rref_rev = B_int.matrix_from_columns(rev_cols).echelon_form()
        rref_fwd = rref_rev.matrix_from_columns(rev_cols)
        ```
     6. Convert each non-zero row of `rref_fwd` via `poly_ring(list(row))` (fast Cython coercion, no `x ** i` loop!) and sort by `key=lambda p: p.degree()`.

6. **`evaluate_message_polynomial(poly_ring: Any, basis: Sequence[Any], message: Sequence[Any]) -> Any`**:
   * Validate `len(message) == len(basis)` (raise `ValueError` otherwise).
   * Return `sum((poly_ring.base_ring()(message[i]) * basis[i] for i in range(len(basis))), poly_ring.zero())`.

7. **`interpolate_erasures(poly_ring: Any, eval_points: Sequence[Any], values: Sequence[Any], erasure_mask: Sequence[Any]) -> list[Any]`**:
   * Collect `known_pts = [(eval_points[i], values[i]) for i in range(len(eval_points)) if int(erasure_mask[i]) == 0]` and `erased_indices = [i for i in range(len(eval_points)) if int(erasure_mask[i]) != 0]`.
   * Copy `out = list(values)`. If `not erased_indices`, return `out`.
   * Compute `interp_poly = poly_ring.lagrange_polynomial(known_pts)`.
   * For `idx` in `erased_indices`: `out[idx] = interp_poly(eval_points[idx])`.
   * Return `out`.

---

## Acceptance Checklist
- [ ] `ruff check src/optimal_lrc/subgroups.py src/optimal_lrc/polynomials.py` passes with zero errors.
- [ ] `ruff format --check src/optimal_lrc/subgroups.py src/optimal_lrc/polynomials.py` passes.
- [ ] Works across prime fields (`GF(13)`, `GF(37)`, `GF(101)`) and extension fields (`GF(16)`, `GF(64)`, `GF(81)`, `GF(121)`, `GF(256)`) regardless of `repr='int'` vs default polynomial representation.
- [ ] Zero wildcard imports, zero bitwise `&` on booleans, zero `int(..., base)` string hacks.
