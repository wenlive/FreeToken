"""Binary GGUF metadata primitives shared by complete and expert-only readers."""
import struct

# Quantized block layouts supported by the complete Flash Next execution path.
GGUF_TYPES = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18),
    6: ("Q5_0", 32, 22), 8: ("Q8_0", 32, 34),
    11: ("Q3_K", 256, 110), 12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210),
    20: ("IQ4_NL", 32, 18), 23: ("IQ4_XS", 256, 136),
    30: ("BF16", 1, 2), 42: ("Q2_0", 64, 18),
}

_SCALARS = {
    0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i",
    6: "f", 7: "?", 10: "Q", 11: "q", 12: "d",
}


def read_exact(stream, count):
    """Reject a short read before a caller interprets metadata or tensor bytes."""
    data = stream.read(count)
    if len(data) != count:
        raise ValueError("truncated GGUF")
    return data


def unpack(stream, fmt):
    layout = struct.Struct("<" + fmt)
    return layout.unpack(read_exact(stream, layout.size))


def read_string(stream):
    size, = unpack(stream, "Q")
    if size > 16 << 20:
        raise ValueError("oversized GGUF string")
    return read_exact(stream, size).decode("utf-8")


def read_value(stream, kind=None, *, keep=True):
    """Skip unneeded tokenizer arrays without allocating their full contents."""
    if kind is None:
        kind, = unpack(stream, "I")
    if kind == 8:
        value = read_string(stream)
        return value if keep else None
    if kind == 9:
        element, count = unpack(stream, "IQ")
        if count > 10_000_000:
            raise ValueError("oversized GGUF metadata array")
        if not keep and element in _SCALARS:
            stream.seek(count * struct.calcsize("<" + _SCALARS[element]), 1)
            return None
        if not keep:
            for _ in range(count):
                read_value(stream, element, keep=False)
            return None
        return [read_value(stream, element) for _ in range(count)]
    if kind not in _SCALARS:
        raise ValueError(f"unsupported GGUF metadata type {kind}")
    return unpack(stream, _SCALARS[kind])[0]
