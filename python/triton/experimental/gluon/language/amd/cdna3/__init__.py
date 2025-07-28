from __future__ import annotations
from typing import TYPE_CHECKING

from triton import knobs
from triton.language import core as tl
from triton.experimental.gluon.language import _core as ttgl
from triton.experimental.gluon.language._core import builtin

if TYPE_CHECKING:
    from ..._semantic import GluonSemantic

__all__ = ["get_amd_mfma_layout", "create_buffer_load", "create_buffer_store", "dot"]


@builtin
def get_amd_mfma_layout(version, tiles_per_warp, warps_per_cta, ctas_per_cga, cta_split_num, cta_order, instr_shape,
                        transposed, elem_type_width, _semantic: GluonSemantic = None):
    return _semantic.builder.get_amd_mfma_layout(version, tiles_per_warp, warps_per_cta, ctas_per_cga, cta_split_num,
                                                 cta_order, instr_shape, transposed, elem_type_width)


@builtin
def create_buffer_load(ptr, element_type, offsets, cache, mask, layout, other, _semantic=None):
    cache_modifier = _semantic._str_to_load_cache_modifier(cache)
    return _semantic.create_buffer_load(ptr.handle, offsets.shape, element_type, offsets.handle, cache_modifier,
                                        mask.handle, other.handle, layout)


@builtin
def create_buffer_store(stored_value, ptr, offsets, cache, mask, _semantic: GluonSemantic = None):
    cache_modifier = _semantic._str_to_load_cache_modifier(cache)

    return _semantic.create_buffer_store(stored_value.handle, ptr.handle, offsets.handle, cache_modifier, mask.handle)


@builtin
def mfma(input, other, acc, layout, input_precision=None, allow_tf32=None, max_num_imprecise_acc=None,
         out_dtype=tl.float32, _semantic: GluonSemantic = None):
    # this is the wrapper of triton.language.dot with one more parameter for layout
    assert acc is not None, "For now, acc is required"
    assert input_precision is None or allow_tf32 is None, "Only one of input_precision and allow_tf32 can be specified"
    if input_precision is None:
        supports_tf32 = "tf32" in _semantic.builder.options.allowed_dot_input_precisions
        input_precision = knobs.language.fp32_default or ("tf32" if (supports_tf32 and
                                                                     (allow_tf32 or allow_tf32 is None)) else "ieee")

    if out_dtype is None:
        out_dtype = acc.dtype
    layout = ttgl._unwrap_if_constexpr(layout)
    input_precision = ttgl._unwrap_if_constexpr(input_precision)
    out_dtype = ttgl._unwrap_if_constexpr(out_dtype)
    max_num_imprecise_acc = ttgl._unwrap_if_constexpr(max_num_imprecise_acc)
    acc = ttgl._unwrap_if_constexpr(acc)

    return _semantic.mfma(input, other, acc, layout, input_precision, max_num_imprecise_acc, out_dtype)
