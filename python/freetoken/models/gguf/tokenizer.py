"""Build a HF fast tokenizer from a GGUF file's embedded tokenizer metadata.

transformers' ``AutoTokenizer.from_pretrained(gguf_file=...)`` first builds the HF
config, which the gemma4 strict dataclass rejects (per-layer ``num_key_value_heads``
array). So we call the GGUF->fast tokenizer converter directly on the
``tokenizer.ggml.*`` metadata, bypassing config entirely.
"""

from __future__ import annotations

from typing import Any

from .reader import gguf_architecture, load_gguf_metadata

# GGUF architecture -> transformers GGUF tokenizer-converter key.
_TOKENIZER_ARCH = {"gemma4": "gemma4_text", "clover_moe": "qwen2"}


def load_gguf_tokenizer(model_path: str):
    from transformers import PreTrainedTokenizerFast
    from transformers.integrations.ggml import convert_gguf_tokenizer

    meta = load_gguf_metadata(model_path)
    arch = meta.get("general.architecture") or gguf_architecture(model_path)
    conv_arch = _TOKENIZER_ARCH.get(arch, arch)
    tok_dict: dict[str, Any] = {
        k[len("tokenizer.ggml.") :]: v
        for k, v in meta.items()
        if k.startswith("tokenizer.ggml.")
    }
    fast, _extra = convert_gguf_tokenizer(conv_arch, tok_dict)
    if arch == "clover_moe":
        from tokenizers import AddedToken, normalizers

        fast.normalizer = (
            normalizers.NFC() if tok_dict.get("normalizer.nfc", False) else None
        )
        # The generic Qwen converter only registers three control tokens.
        # Clover also needs literal reasoning/tool/mask token recognition.
        for token, kind in zip(tok_dict["tokens"], tok_dict["token_type"], strict=True):
            if kind in (3, 4):  # GGML CONTROL, USER_DEFINED
                added = AddedToken(token, normalized=False, special=kind == 3)
                if kind == 3:
                    fast.add_special_tokens([added])
                else:
                    fast.add_tokens([added])

    tokens = tok_dict["tokens"]

    def tok_for(id_key: str, default: str) -> str:
        tid = meta.get(f"tokenizer.ggml.{id_key}")
        return (
            tokens[int(tid)] if tid is not None and int(tid) < len(tokens) else default
        )

    # gemma4 chat turns end with <turn|>; prefer it as eos so chat generation halts
    # (the formal <eos> is also a stop id, see gguf_eos_token_ids).
    turn_end = "<turn|>" if "<turn|>" in tokens else None
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=fast,
        bos_token=tok_for("bos_token_id", "<bos>"),
        eos_token=turn_end or tok_for("eos_token_id", "<eos>"),
        # Qwen/Clover has no unknown token; adding <unk> would grow the vocab.
        unk_token=tok_for("unknown_token_id", "<unk>")
        if arch != "clover_moe"
        else None,
        pad_token=tok_for("padding_token_id", "<pad>"),
        **(
            {"mask_token": tok_for("mask_token_id", "<|mask|>")}
            if arch == "clover_moe"
            else {}
        ),
    )
    chat_template = meta.get("tokenizer.chat_template")
    if chat_template:
        tokenizer.chat_template = chat_template
    return tokenizer


def gguf_eos_token_ids(model_path: str, tokenizer) -> set[int]:
    """Stop ids for GGUF generation: the formal <eos> plus the chat turn end <turn|>."""
    meta = load_gguf_metadata(model_path)
    tokens = meta["tokenizer.ggml.tokens"]
    ids: set[int] = set()
    if tokenizer.eos_token_id is not None:
        ids.add(int(tokenizer.eos_token_id))
    eid = meta.get("tokenizer.ggml.eos_token_id")
    if eid is not None:
        ids.add(int(eid))
    # Look the stop tokens up in the vocab directly (convert_tokens_to_ids would map an
    # absent name to <unk>, wrongly adding it as a stop id).
    for name in ("<eos>", "<turn|>"):
        try:
            ids.add(tokens.index(name))
        except ValueError:
            pass
    if (meta.get("general.architecture") or gguf_architecture(model_path)) == "clover_moe":
        # Clover's published generation config stops on both im_end and endoftext.
        ids.add(int(meta["tokenizer.ggml.padding_token_id"]))
    return ids


__all__ = ["load_gguf_tokenizer", "gguf_eos_token_ids"]
