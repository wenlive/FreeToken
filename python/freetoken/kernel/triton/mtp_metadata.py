"""一次 H2D 输入包与一个 kernel 更新单请求验证的原 graph 元数据。"""
import triton
import triton.language as tl


@triton.jit
def _prepare(PACKET, PT, INPUT, POS, LOC, CACHE, LINEAR, RING, LENS, TABLE,
             PTSTRIDE: tl.constexpr, PAGESIZE: tl.constexpr, WIDTH: tl.constexpr,
             T: tl.constexpr, BLOCK: tl.constexpr):
    col = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    position = tl.load(PACKET)
    owner = tl.load(PACKET + 1).to(tl.int64)
    slot = tl.load(PACKET + 2)
    token = tl.load(PACKET + 4 + col, col < T, 0)
    p = position + col
    loc = tl.load(PT + owner * PTSTRIDE + p, col < T, 0)
    tl.store(INPUT + col, token, col < T)
    tl.store(POS + col, p, col < T)
    tl.store(LOC + col, loc, col < T)
    if tl.program_id(0) == 0:
        tl.store(CACHE, slot)
        tl.store(LINEAR, slot)
        tl.store(RING, owner)
        tl.store(LENS, position + T)
    page = tl.load(PT + owner * PTSTRIDE + col * PAGESIZE, col < WIDTH, 0) // PAGESIZE
    tl.store(TABLE + col, page, col < WIDTH)


def stage_verify_metadata(packet, page_table, batch, page_size):
    """只更新既有 graph buffer；PLE 与 graph 的消费顺序由 MtpRuntime 保持。"""
    md = batch.attn_metadata
    width = md.block_table.numel()
    _prepare[(triton.cdiv(width, 256),)](packet, page_table, batch.input_ids, batch.positions,
        batch.out_loc, batch.fla_metadata.cache_indices, batch.linear_table_idx,
        md.ring_slots, md.seq_lens, md.block_table, page_table.stride(0), page_size,
        width, len(batch.input_ids), 256, num_warps=4)
