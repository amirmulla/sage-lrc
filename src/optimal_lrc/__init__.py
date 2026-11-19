"""Optimal Locally Recoverable Codes (single-set and two-set Tamo-Barg codes) in SageMath."""

import importlib
from typing import TYPE_CHECKING, Any

# Initialize SageMath's Cython coercion and category framework once on package import.
# Supports both monolithic SageMath (`sage -python`) and modular `passagemath` wheels.
try:
    import sage.all  # noqa: F401
except ModuleNotFoundError:
    import sage.all__sagemath_modules  # noqa: F401

if TYPE_CHECKING:
    from optimal_lrc.expander import (
        compute_expansion_metrics,
        create_bipartite_coset_graph,
        get_error_position_subgraph,
        save_bipartite_graph_plot,
    )
    from optimal_lrc.subgroups import (
        field_element_to_int,
        find_additive_subgroups,
        find_additive_subgroups_bruteforce,
        find_subspaces,
        is_closed_under_addition,
        list_canonical_additive_subgroups,
        list_multiplicative_subgroups,
        list_subgroup_cosets,
        shift_additive_subgroup,
        sort_field_elements,
    )
    from optimal_lrc.tamo_barg import (
        TamoBargCode,
        TamoBargErasureDecoder,
        TamoBargErrorDecoder,
        TamoBargVectorEncoder,
    )
    from optimal_lrc.tamo_barg_two_sets import (
        TamoBargCodeTwoSets,
        TamoBargGlobalDecoder,
        TamoBargGlobalEEDecoder,
        TamoBargIterativeDecoder,
        TamoBargIterativeEEDecoder,
        TamoBargTwoSetsVectorEncoder,
        TamoBargTwoStepDecoder,
        TamoBargTwoStepEEDecoder,
    )

__version__ = "0.2.0"

__all__ = [
    "TamoBargCode",
    "TamoBargVectorEncoder",
    "TamoBargErasureDecoder",
    "TamoBargErrorDecoder",
    "TamoBargCodeTwoSets",
    "TamoBargTwoSetsVectorEncoder",
    "TamoBargIterativeDecoder",
    "TamoBargIterativeEEDecoder",
    "TamoBargGlobalDecoder",
    "TamoBargGlobalEEDecoder",
    "TamoBargTwoStepDecoder",
    "TamoBargTwoStepEEDecoder",
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
    "create_bipartite_coset_graph",
    "get_error_position_subgraph",
    "compute_expansion_metrics",
    "save_bipartite_graph_plot",
]

_EXPORT_TO_MODULE: dict[str, str] = {
    # optimal_lrc.tamo_barg
    "TamoBargCode": "optimal_lrc.tamo_barg",
    "TamoBargVectorEncoder": "optimal_lrc.tamo_barg",
    "TamoBargErasureDecoder": "optimal_lrc.tamo_barg",
    "TamoBargErrorDecoder": "optimal_lrc.tamo_barg",
    # optimal_lrc.tamo_barg_two_sets
    "TamoBargCodeTwoSets": "optimal_lrc.tamo_barg_two_sets",
    "TamoBargTwoSetsVectorEncoder": "optimal_lrc.tamo_barg_two_sets",
    "TamoBargIterativeDecoder": "optimal_lrc.tamo_barg_two_sets",
    "TamoBargIterativeEEDecoder": "optimal_lrc.tamo_barg_two_sets",
    "TamoBargGlobalDecoder": "optimal_lrc.tamo_barg_two_sets",
    "TamoBargGlobalEEDecoder": "optimal_lrc.tamo_barg_two_sets",
    "TamoBargTwoStepDecoder": "optimal_lrc.tamo_barg_two_sets",
    "TamoBargTwoStepEEDecoder": "optimal_lrc.tamo_barg_two_sets",
    # optimal_lrc.subgroups
    "field_element_to_int": "optimal_lrc.subgroups",
    "sort_field_elements": "optimal_lrc.subgroups",
    "is_closed_under_addition": "optimal_lrc.subgroups",
    "find_subspaces": "optimal_lrc.subgroups",
    "find_additive_subgroups": "optimal_lrc.subgroups",
    "find_additive_subgroups_bruteforce": "optimal_lrc.subgroups",
    "list_multiplicative_subgroups": "optimal_lrc.subgroups",
    "list_canonical_additive_subgroups": "optimal_lrc.subgroups",
    "shift_additive_subgroup": "optimal_lrc.subgroups",
    "list_subgroup_cosets": "optimal_lrc.subgroups",
    # optimal_lrc.expander
    "create_bipartite_coset_graph": "optimal_lrc.expander",
    "get_error_position_subgraph": "optimal_lrc.expander",
    "compute_expansion_metrics": "optimal_lrc.expander",
    "save_bipartite_graph_plot": "optimal_lrc.expander",
}


def __getattr__(name: str) -> Any:
    """Lazily resolve and cache public package exports per PEP 562."""
    module_name = _EXPORT_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return sorted module attributes including lazy PEP 562 exports."""
    return sorted({*globals().keys(), *__all__})
