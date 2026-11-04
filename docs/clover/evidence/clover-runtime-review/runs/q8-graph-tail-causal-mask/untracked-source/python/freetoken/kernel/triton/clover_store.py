"""Clover verify stores only owned, valid prefix rows of a padded block."""

import triton
import triton.language as tl


@triton.jit
def _store(KC, VC, LOC, K, V, VALID, KS: tl.constexpr, VS: tl.constexpr,
           FEATURES: tl.constexpr, TILE: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1) * TILE + tl.arange(0, TILE)
    valid = tl.load(VALID + row)
    loc = tl.load(LOC + row, valid, other=0).to(tl.int64)
    k = tl.load(K + row * KS + col, valid & (col < FEATURES), other=0)
    v = tl.load(V + row * VS + col, valid & (col < FEATURES), other=0)
    tl.store(KC + loc * FEATURES + col, k, valid & (col < FEATURES))
    tl.store(VC + loc * FEATURES + col, v, valid & (col < FEATURES))


def store_verify_kv(keys, values, locations, k, v, valid):
    rows, features = k.shape[0], keys.shape[1] * keys.shape[2]
    k, v = k.reshape(rows, features), v.reshape(rows, features)
    assert k.stride(1) == v.stride(1) == 1
    _store[(rows, triton.cdiv(features, 256))](keys, values, locations, k, v, valid,
        k.stride(0), v.stride(0), features, 256)
