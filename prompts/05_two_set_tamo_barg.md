# Agent 05 — Two-Set Tamo–Barg Code, Encoder & Decoders Specialist (Wave 2)

## Role & Mission
You are the **Two-Set Tamo–Barg Code, Encoder & Decoders Specialist** in Wave 2 of the `tamo_barg_lrc` refactoring swarm. First read [`prompts/shared_context_and_contracts.md`](./shared_context_and_contracts.md) completely, then inspect `src/TamoBargTwoSets.py`, `src/tamo_barg_lrc/subgroups.py`, `src/tamo_barg_lrc/polynomials.py`, and `src/tamo_barg_lrc/expander.py`.

## Exclusive File Ownership
You may **ONLY** create or modify:
* `src/tamo_barg_lrc/tamo_barg_two_sets.py`

Do **NOT** delete legacy `src/TamoBargTwoSets.py` (Agent 07 deletes it in Wave 3) and do **NOT** touch any other files.

---

## Bugs & Performance Targets Assigned to You
* **B1 (CRITICAL)**: In `TamoBargTwoSets.py:452` and `:546`, `if (uncorrectable_err[0] == False & uncorrectable_err[1] == False):` evaluates `False & uncorrectable_err[1]` (`False`) before `==`, turning into `uncorrectable_err[0] == False == False` and **completely ignoring Partition 1 errors**! Replace with `not uncorrectable_err[0] and not uncorrectable_err[1]` plus full-partition syndrome verification.
* **B2 (CRITICAL)**: In `TamoBargIterativeEEDecoder.decode_to_code` (`TamoBargTwoSets.py:515-539`), when `erasure_weight == d_local - 1`, `ErrorErasure` decoder is called *first*, throws an exception appending `due = True` and setting `uncorrectable_err[i] = True`, and *then* Lagrange interpolation recovers the erasures without resetting `due` or `uncorrectable_err`! Check `erasure_weight == d_local - 1` **before** invoking `ErrorErasure` so recovered cosets record `due = False`.
* **B3 (CRITICAL)**: Fix `self.code().evaluation_ponts()` typo in `TamoBargGlobalEEDecoder.decode_to_code` (`TamoBargTwoSets.py:603`) $\to$ `self.code().evaluation_points()`.
* **B4**: Never mutate input vectors (`r`, `word`, `erasure_vector`) in-place inside any decoder (`work = copy(vector(self._F, r))`).
* **B5**: Fix `TamoBargGlobalEEDecoder.__init__` `input_space` to `cartesian_product([code.ambient_space(), VectorSpace(GF(2), code.length())])`.
* **B6**: Implement `enc_poly(self, message)` on `TamoBargTwoSetsVectorEncoder` (fixing undefined `self._enc_poly`).
* **B9**: Fix `shift_add` subgroup shifting bug (`TamoBargTwoSets.py:102-110`): when `sub_group_type == ["add", "add"]` and `shift_add=True`, legacy code shifted *both* `i=0` and `i=1` by `a**t_0`, making them overlap completely!
* **B10**: Remove the broken dead `num_of_sets == 1` branch in `TamoBargCodeTwoSets` (require 2 sets) and fix `["mult", "add"]` `ValueError: 0 is not in list` by building `self._evaluation_points` as the canonically sorted union of evaluation points across both partitions, using an $O(n)$ dict `{pt: idx}` instead of $O(n^2)$ `.index(h)`.
* **B11**: Raise descriptive `ValueError` exceptions in `__init__` for invalid divisibility, missing subgroups, or `dimension > max_dimension` instead of `print(...)` + `AttributeError`.
* **B13**: Rename `TamoBargIterariveDecoder` $\to$ `TamoBargIterativeDecoder`, `TamoBargIterariveEEDecoder` $\to$ `TamoBargIterativeEEDecoder`, `_parition_size` $\to$ `_partition_size`, and eliminate all boolean `&` operators.
* **B17**: Replace all bare `except:` clauses with `except (DecodingError, ValueError):`.
* **P1 & P2**: Precompute local GRS parity-check matrices $H_{i,j}$ for every coset in both partitions to skip zero-syndrome cosets in $O(m)$ Cython/FLINT arithmetic, and add exact convergence + trapping-set stall detection.
* **P4**: Delegate bipartite graph construction to `create_bipartite_coset_graph` ($O(n)$).
* **P5**: Delegate degree-minimizing basis intersection (`intersect_encoding_bases`) to `tamo_barg_lrc.polynomials`.
* **P6**: Speed up `minimum_distance_py()` by $(q - 1)\times$ by iterating only over monic (projectively normalized, first non-zero coordinate $= 1$) messages in $GF(q)^k$.

---

## Detailed Implementation Specifications: `src/tamo_barg_lrc/tamo_barg_two_sets.py`

### 1. `TamoBargCodeTwoSets(AbstractLinearCode)`
* **Signature**:
  ```python
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
  ```
* **Validation & Normalization**:
  1. Require `base_field.is_finite()`.
  2. Require `isinstance(locality, (list, tuple)) and len(locality) == 2` (else raise `ValueError("locality must be a 2-element sequence [r1, r2]")`). Store `self._num_of_sets = 2` and `self._locality = [int(locality[0]), int(locality[1])]`.
  3. Normalize `local_minimum_distance`: if `int`, expand to `[d, d]`; else require length 2. Store `self._local_minimum_distance = [int(d0), int(d1)]`.
  4. Normalize `sub_group_type` to a 2-element list with elements in `("add", "mult")`.
  5. Normalize `subgroup` to `[None, None]` if `None`, else a 2-element list.
* **Subgroup & Coset Partition Construction (Fixes B9, B10, B11)**:
  1. Compute `coset_sizes = [self._locality[i] + self._local_minimum_distance[i] - 1 for i in range(2)]`.
  2. Enumerate `mult_subgroups = list_multiplicative_subgroups(base_field)` and `add_subgroups = list_canonical_additive_subgroups(base_field)`.
  3. For `i` in `(0, 1)`:
     * If `subgroup[i] is not None`:
       * Validate `len(subgroup[i]) == coset_sizes[i]` and store `sort_field_elements(subgroup[i])`.
     * Else if `self._sub_group_type[i] == "mult"`:
       * Require `coset_sizes[i] in mult_subgroups`, else raise `ValueError`.
       * Store `mult_subgroups[coset_sizes[i]]`.
     * Else (`self._sub_group_type[i] == "add"`):
       * Require `coset_sizes[i] in add_subgroups`, else raise `ValueError`.
       * Start with `sg = add_subgroups[coset_sizes[i]]`.
       * **Shifting rule (Fixes B9)**:
         * Let `p = int(base_field.characteristic())` and `t_0` be the integer $\log_p$ of `len(self._sub_group[0])` when `i == 1` (or `int_log_p(len(sg))` for mixed mode).
         * If `self._sub_group_type == ["add", "add"]`:
           * For `i == 0`: if `shift_add` is `True`, shift by `1`; else do not shift.
           * For `i == 1`: shift by `(1 + t_0) if shift_add else t_0` via `shift_additive_subgroup(base_field, sg, shift_exp)` so `sub_group[0]` and `sub_group[1]` never get shifted by the same exponent!
         * Else (mixed `["add", "mult"]` or `["mult", "add"]`):
           * If `shift_add` is `True`, shift the `"add"` subgroup by `int_log_p(len(sg))`.
       * Store `sg`.
  4. **Divisibility & Partition Size Validation (Supports `["add", "add"]`, `["mult", "mult"]`, `["add", "mult"]`, and `["mult", "add"]`!)**:
     * Let `q = int(base_field.cardinality())` and `has_add = "add" in self._sub_group_type`.
     * For `i` in `(0, 1)`:
       * If `self._sub_group_type[i] == "add"`:
         * Require `length <= q` and `length % coset_sizes[i] == 0` (else raise `ValueError`).
         * `p_size = length // coset_sizes[i]`
       * Else (`"mult"`):
         * If `has_add and length == q`:
           * Multiplicative cosets partition $GF(q)^\times$ of size $q - 1$; require `(q - 1) % coset_sizes[i] == 0` (else raise `ValueError`).
           * `p_size = (q - 1) // coset_sizes[i]`
         * Else:
           * Require `length <= q - 1` and `length % coset_sizes[i] == 0` (else raise `ValueError`).
           * `p_size = length // coset_sizes[i]`
       * `self._partition_size.append(p_size)`
       * `self._partition.append(list_subgroup_cosets(base_field, self._sub_group[i], self._sub_group_type[i])[:p_size])`
  5. **Canonical Unified Evaluation Points & $O(n)$ Index Map (Fixes B10 & P4)**:
     * `all_pts = {pt for part in self._partition for coset in part for pt in coset}`
     * `self._evaluation_points = sort_field_elements(all_pts)`
     * Require `len(self._evaluation_points) == length` (else raise `ValueError`).
     * `pt_to_idx = {pt: idx for idx, pt in enumerate(self._evaluation_points)}`
     * `self._evaluation_points_idx = [[[pt_to_idx[h] for h in coset] for coset in part] for part in self._partition]`
  6. **Bipartite Graph & Encoder Dimension Validation**:
     * `self._G, self._G_L, self._G_R = create_bipartite_coset_graph(self._partition, self._evaluation_points_idx, self._evaluation_points)`
     * Call `super().__init__(base_field, int(length), "VectorEncoder", "IterativeDecoder")` and set `self._dimension = int(dimension)`.
     * Validate `1 <= self._dimension <= self.max_dimension()` (raise `ValueError` if `self._dimension < 1` or `self._dimension > self.max_dimension()`).

* **Code Methods**:
  * `bipartite_graph()`, `partition()`, `sub_group()`, `evaluation_points_idx()`, `evaluation_points()`, `locality()`, `num_of_sets()`, `local_minimum_distance()`, `max_dimension()`, `max_rate()`.
  * `design_distance(self) -> int`:
    ```python
    basis = self.encoder().enc_basis()
    max_degree = max((poly.degree() for poly in basis), default=0)
    return int(self.length() - max_degree)
    ```
  * **`minimum_distance_py(self) -> int` (Fixes P6 — $(q - 1)\times$ Projective Speedup)**:
    Iterate only over normalized (monic) message vectors in $GF(q)^k$ whose first non-zero coordinate is `1`:
    ```python
    gen_mat = self.generator_matrix()
    field = self.base_field()
    k = self.dimension()
    min_wt = int(self.length())
    zero = field.zero()
    one = field.one()
    elems = list(field)

    # Pivot at coordinate `lead` (0 <= lead < k): msg = (0, ..., 0, 1, *, ..., *)
    from itertools import product

    for lead in range(k):
        prefix = [zero] * lead + [one]
        for tail in product(elems, repeat=k - lead - 1):
            msg = vector(field, prefix + list(tail))
            cw = msg * gen_mat
            wt = int(cw.hamming_weight())
            if 0 < wt < min_wt:
                min_wt = wt
                if min_wt == 1:
                    return 1
    return min_wt
    ```

---

### 2. `TamoBargTwoSetsVectorEncoder(Encoder)`
* Compute per-partition `_good_poly[i]`, `_algebra_basis[i]`, `_enc_basis[i]` via `tamo_barg_lrc.polynomials` over `PolynomialRing(code.base_field(), "x")` (no `PolynomialQuotientRing` needed!).
* Compute degree-minimizing intersection basis `full_comb_basis = intersect_encoding_bases(self._base_poly_ring, self._enc_basis[0], self._enc_basis[1], code.length())` (**Fixes P5**).
* Store `self._max_dimension = len(full_comb_basis)` and `self._comb_enc_basis = full_comb_basis[:code.dimension()]`.
* Implement `enc_basis()`, `enc_basis_ext()`, `enc_algebra_basis()`, `enc_good_poly()`, `max_dimension()`, `enc_poly(self, message)` (**Fixes B6**), and `@cached_method def generator_matrix(self)`.

---

### 3. Decoders (Strictly Preserving Legacy Tuple Return Signatures!)

#### A. `TamoBargIterativeDecoder(Decoder)` (Registered as `"IterativeDecoder"`)
* `__init__(self, code, max_num_of_itr: int = 10)`:
  * Precompute for both partitions `i in (0, 1)` and each `coset`:
    * `local_grs = GeneralizedReedSolomonCode(coset, code.locality()[i])`
    * `local_dec = local_grs.decoder("Gao")`
    * `local_pcm = local_grs.parity_check_matrix()` (**Fixes P1**)
* `decode_to_code(self, r) -> tuple[Any, int]`:
  1. `work = copy(vector(self._F, r))` (**Fixes B4**).
  2. Loop `while num_itr < self._max_num_of_itr:`:
     * `uncorrectable_err = [False, False]`
     * `changed_in_itr = False`
     * `changed_in_p1 = False`
     * For `i` in `(0, 1)`:
       * For `j, idxs in enumerate(self._evalpts_idx[i])`:
         * `local_vec = vector(self._F, [work[idx] for idx in idxs])`
         * **Fast syndrome check**: `if (self._local_parity_checks[i][j] * local_vec).is_zero(): continue`
         * Try `corr_c = self._partitions_local_decoders[i][j].decode_to_code(local_vec)` catching `(DecodingError, ValueError)` (**Fixes B17**):
           * On success: write back `work[idx] = corr_c[pos]`; if `corr_c != local_vec`, set `changed_in_itr = True` (and if `i == 1`, `changed_in_p1 = True`).
           * On failure: `uncorrectable_err[i] = True`.
     * `num_itr += 1`
     * **Convergence & stall checks (Fixes B1 & P2)**:
       * If `not uncorrectable_err[0] and not uncorrectable_err[1]`:
         * If `not changed_in_p1` or `self._partition_is_syndrome_clean(0, work)`:
           `break`
       * Else if `not changed_in_itr`:
         `break` (trapping-set fixed point—further iterations cannot change `work`!)
  3. Return `work, num_itr`.

#### B. `TamoBargIterativeEEDecoder(Decoder)` (Registered as `"IterativeErasureErrorDecoder"`)
* `__init__(self, code, max_num_of_itr: int = 10, erasure_start_itr: int | None = None)`:
  * Default `self._erasure_start_itr = max_num_of_itr // 2 if erasure_start_itr is None else erasure_start_itr`.
  * Precompute `self._poly_ring = PolynomialRing(code.base_field(), "x")`, local GRS codes, `"ErrorErasure"` decoders, and local parity-check matrices $H_{i,j}$.
* `decode_to_code(self, r) -> tuple[tuple[Any, Any], int]`:
  1. `work = copy(vector(self._F, r))` (**Fixes B4**), `erasure_flags = [0] * len(work)`.
  2. Loop `while num_itr < self._max_num_of_itr:`:
     * Snapshot `prev_work = list(work)` and `prev_erasures = list(erasure_flags)`.
     * `uncorrectable_err = [False, False]`
     * `due_status: list[list[bool]] = []`
     * For `i` in `(0, 1)`:
       * `d_local = self._local_min_distances[i]`
       * `due_tmp: list[bool] = []`
       * For `j, coset in enumerate(self._partitions[i])`:
         * `idxs = self._evalpts_idx[i][j]`
         * `r_list = [work[idx] for idx in idxs]`
         * `era_list = [erasure_flags[idx] for idx in idxs]`
         * `era_wt = sum(era_list)`
         * **Branch order (Fixes B2 & P1)**:
           1. `if era_wt == 0 and (self._local_parity_checks[i][j] * vector(self._F, r_list)).is_zero():`
              `due_tmp.append(False)`
           2. `elif era_wt >= d_local:`
              `due_tmp.append(True); uncorrectable_err[i] = True`
           3. `elif era_wt == d_local - 1:`
              `corr_list = interpolate_erasures(self._poly_ring, coset, r_list, era_list)`
              write `corr_list` into `work`; `due_tmp.append(False)` (**recovered—NOT DUE!**)
           4. `else:`
              try `corr_c = self._partitions_local_decoders[i][j].decode_to_code((vector(self._F, r_list), vector(GF(2), era_list)))` catching `(DecodingError, ValueError)`:
              on success write `corr_c` into `work` and `due_tmp.append(False)`; on failure `due_tmp.append(True); uncorrectable_err[i] = True`.
       * `due_status.append(due_tmp)`
     * `num_itr += 1`
     * If `not uncorrectable_err[0] and not uncorrectable_err[1]` and `self._partition_is_syndrome_clean(0, work)`:
       * Clear `erasure_flags` on all cosets with `due_status[s][t] is False` and `break`!
     * If `num_itr >= self._erasure_start_itr`:
       * For `t, idxs in enumerate(self._evalpts_idx[0])`:
         `mark = 1 if due_status[0][t] else 0`
         `for idx in idxs: erasure_flags[idx] = mark`
       * For `t, idxs in enumerate(self._evalpts_idx[1])`:
         `if not due_status[1][t]:`
           `for idx in idxs: erasure_flags[idx] = 0`
     * If `num_itr > self._erasure_start_itr and list(work) == prev_work and erasure_flags == prev_erasures`:
       `break`
  3. Return `(work, vector(GF(2), erasure_flags)), num_itr`.

#### C. `TamoBargGlobalDecoder(Decoder)` (Registered as `"GlobalDecoder"`)
* `__init__(self, code)`: builds `GeneralizedReedSolomonCode(code.evaluation_points(), code.length() - code.design_distance() + 1)` and its `"Gao"` decoder + parity-check matrix.
* `decode_to_code(self, r) -> Vector`: clones `r`, short-circuits if global syndrome is zero, otherwise calls `self._global_decoder.decode_to_code(work)`.

#### D. `TamoBargGlobalEEDecoder(Decoder)` (Registered as `"GlobalErasureErrorDecoder"`)
* `__init__(self, code)`: sets `input_space = cartesian_product([code.ambient_space(), VectorSpace(GF(2), code.length())])` (**Fixes B5**), precomputes `_global_grs`, `"ErrorErasure"` decoder, global parity-check matrix, and `_poly_ring`.
* `decode_to_code(self, word_and_erasure_vector) -> Vector`:
  * Unpacks `word, erasure_vector = word_and_erasure_vector` without mutating `word` (**Fixes B4**).
  * `era_wt = int(erasure_vector.hamming_weight())`, `d_glob = int(self._global_grs.minimum_distance())`.
  * If `era_wt == 0 and (self._global_pcm * vector(self._F, word)).is_zero()`: return `copy(vector(self._F, word))`.
  * If `era_wt >= d_glob`: raise `DecodingError("Too many erasures for global decoder")`.
  * If `era_wt == d_glob - 1`: use `self.code().evaluation_points()` (**Fixes B3**) and `interpolate_erasures(...)`, returning `vector(self._F, corr_list)`.
  * Else: return `self._global_decoder.decode_to_code((copy(vector(self._F, word)), vector(GF(2), erasure_vector)))`.

#### E. `TamoBargTwoStepDecoder(Decoder)` (Registered as `"TwoStepsDecoder"`)
* Runs `tmp, num_itr = self._iter_decoder.decode_to_code(r)`, then runs `self._global_decoder.decode_to_code(tmp)` catching `(DecodingError, ValueError)` (falling back to `tmp`), and returns `(result, num_itr)`.

#### F. `TamoBargTwoStepEEDecoder(Decoder)` (Registered as `"TwoStepsErasureErrorDecoder"`)
* Runs `(word_after_iter, era_vec), num_itr = self._iter_ee_decoder.decode_to_code(r)`, computes `num_rem_era = int(era_vec.hamming_weight())`, runs `self._global_ee_decoder.decode_to_code((word_after_iter, era_vec))` catching `(DecodingError, ValueError)` (falling back to `word_after_iter`), and returns `(result, num_itr, num_rem_era)`.

---

## Acceptance Checklist
- [ ] `ruff check src/tamo_barg_lrc/tamo_barg_two_sets.py` and `ruff format --check src/tamo_barg_lrc/tamo_barg_two_sets.py` pass cleanly.
- [ ] All 6 decoder keys (`"IterativeDecoder"`, `"TwoStepsDecoder"`, `"GlobalDecoder"`, `"IterativeErasureErrorDecoder"`, `"GlobalErasureErrorDecoder"`, `"TwoStepsErasureErrorDecoder"`) match `shared_context_and_contracts.md` Section 4.
- [ ] Both `["add", "mult"]` AND `["mult", "add"]` construct cleanly without `ValueError: 0 is not in list` (**B10**).
- [ ] Partition-1-only uncorrectable errors do NOT prematurely stop iteration 1 (**B1**).
- [ ] Local and global `era_wt == d - 1` erasure recovery succeeds without `AttributeError` (**B3**) or false `DUE` marking (**B2**).
