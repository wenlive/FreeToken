// Source: Maphist0/llama.cpp-pub@cb09d4355d8bf968bb907c5de03cf42d8aa073ce,
// examples/speculative-simple/speculative-simple.cpp and common/speculative.cpp.
// MIT; Copyright (c) 2023-2026 the llama.cpp authors. See LICENSE.clover-gguf-reference.
// Local harness only: call the stock draft-self implementation and read llama_perf.
// No llama.cpp algorithm, source file or library is modified.
#include "arg.h"
#include "common.h"
#include "speculative.h"
#include "llama.h"
#include "ggml.h"

#include <algorithm>
#include <cstdio>
#include <stdexcept>

int main(int argc, char ** argv) {
    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SPECULATIVE)) return 1;
    const auto limits = common_speculative_get_output_limits(
        params.n_batch, params.n_parallel, common_speculative_n_max(&params.speculative));
    params.n_outputs_max = limits.total;
    params.n_outputs_max_per_seq = limits.per_seq;
    llama_backend_init();
    llama_numa_init(params.numa);
    auto initialized = common_init_from_params(params);
    auto * ctx = initialized->context();
    if (!ctx) throw std::runtime_error("stock context initialization failed");
    params.speculative.draft.ctx_tgt = ctx;
    common_speculative_ptr spec(common_speculative_init(params.speculative, 1));
    if (!spec) throw std::runtime_error("stock speculator initialization failed");
    const auto input = common_tokenize(ctx, params.prompt, true, true);
    if (input.size() < 2) throw std::runtime_error("prompt is too short");
    common_batch prefix(ctx);
    for (size_t i = 0; i + 1 < input.size(); ++i) prefix.add(input[i], i, 0, false);
    if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, prefix.get())) return 2;
    if (!common_speculative_process(spec.get(), prefix)) return 3;
    llama_tokens history(input.begin(), input.end() - 1), draft;
    common_speculative_begin(spec.get(), 0, history);
    auto & dp = common_speculative_get_draft_params(spec.get(), 0);
    dp.drafting = true;
    dp.n_max = params.speculative.draft.n_max;
    dp.pos0 = history.size();
    dp.id_last = input.back();
    dp.prompt = &history;
    dp.result = &draft;
    common_speculative_draft(spec.get());
    if (!draft.empty()) throw std::runtime_error("expected the stock first-anchor guard");
    common_batch anchor(ctx);
    anchor.add(input.back(), history.size(), 0, true);
    if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, anchor.get())) return 4;
    const float * logits = llama_get_logits_ith(ctx, -1);
    const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(initialized->model()));
    const llama_token next = std::max_element(logits, logits + vocab) - logits;
    history.push_back(input.back());
    std::printf("{\"type\":\"input\",\"anchor\":%d,\"ids\":[", next);
    for (size_t i = 0; i < input.size(); ++i) std::printf("%s%d", i ? "," : "", input[i]);
    std::printf("]}\n");
    for (int repeat = 0; repeat < 7; ++repeat) {
        draft.clear();
        dp.drafting = true;
        dp.n_max = params.speculative.draft.n_max;
        dp.pos0 = history.size();
        dp.id_last = next;
        llama_perf_context_reset(ctx);
        const auto start = ggml_time_us();
        common_speculative_draft(spec.get());
        llama_synchronize(ctx);
        const double wall_ms = (ggml_time_us() - start) / 1000.;
        const auto perf = llama_perf_context(ctx);
        if (draft.size() != size_t(params.speculative.draft.n_max))
            throw std::runtime_error("stock self-draft did not produce its configured width");
        std::printf("{\"type\":\"preview\",\"width\":%d,\"repeat\":%d,\"warmup\":%s,"
                    "\"decode_ms\":%.6f,\"wall_ms\":%.6f,\"prompt_eval_tokens\":%d,\"eval_tokens\":%d}\n",
            params.speculative.draft.n_max + 1, repeat, repeat < 2 ? "true" : "false",
            perf.t_p_eval_ms + perf.t_eval_ms, wall_ms, perf.n_p_eval, perf.n_eval);
        std::fflush(stdout);
    }
    return 0;
}
