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
#include <cstring>
#include <filesystem>
#include <fstream>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

// The stock non-causal decode accepts arbitrary block tokens, including fixed rows.
// Keep this numerical path separate from the unchanged draft-self timing loop below.
static void dump_cases(llama_context * ctx, const std::string & input_path, const std::string & output_path) {
    std::ifstream input(input_path);
    input.exceptions(std::ios::badbit | std::ios::failbit);
    const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
    const size_t row_bytes = size_t(vocab) * sizeof(float);
    auto read_tokens = [&](int count) {
        if (count < 1 || count > int(llama_n_ctx_seq(ctx))) throw std::runtime_error("invalid input length");
        llama_tokens tokens(count);
        for (auto & token : tokens) {
            input >> token;
            if (token < 0 || token >= vocab) throw std::runtime_error("invalid token id");
        }
        return tokens;
    };
    auto output_file = [&](const std::string & name) {
        if (std::filesystem::exists(std::filesystem::path(output_path) / name))
            throw std::runtime_error("diagnostic output already exists");
        std::ofstream file(std::filesystem::path(output_path) / name, std::ios::binary);
        file.exceptions(std::ios::badbit | std::ios::failbit);
        return file;
    };
    auto write_row = [&](std::ofstream & file, int row) {
        const auto * logits = llama_get_logits_ith(ctx, row);
        if (!logits) throw std::runtime_error("missing output logits");
        file.write(reinterpret_cast<const char *>(logits), row_bytes);
    };
    int cases;
    input >> cases;
    if (cases < 1) throw std::runtime_error("no diagnostic cases");
    for (int c = 0; c < cases; ++c) {
        int prefix_len, steps, blocks;
        input >> prefix_len >> steps >> blocks;
        if (steps < 2 || blocks < 1 || prefix_len + steps + 32 >= int(llama_n_ctx_seq(ctx)))
            throw std::runtime_error("invalid diagnostic dimensions");
        const auto prefix_ids = read_tokens(prefix_len);
        const auto forced = read_tokens(steps);
        std::vector<llama_tokens> block_ids;
        for (int b = 0; b < blocks; ++b) {
            int width;
            input >> width;
            if (width != 8 && width != 16 && width != 32) throw std::runtime_error("invalid block width");
            block_ids.push_back(read_tokens(width));
        }
        llama_memory_clear(llama_get_memory(ctx), true);
        common_batch batch(ctx);
        for (int i = 0; i < prefix_len; ++i) batch.add(prefix_ids[i], i, 0, i + 1 == prefix_len);
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()))
            throw std::runtime_error("diagnostic prefill failed");
        common_prompt_checkpoint clean;
        clean.update_tgt(ctx, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        auto restore = [&]() {
            if (!clean.load_tgt(ctx, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) ||
                    !llama_memory_seq_rm(llama_get_memory(ctx), 0, prefix_len, -1))
                throw std::runtime_error("diagnostic state restoration failed");
        };
        auto causal = output_file("case-" + std::to_string(c) + "-causal.f32");
        std::vector<float> second_row;
        for (int step = 0; step < steps; ++step) {
            write_row(causal, -1);
            if (step == 1) {
                const auto * logits = llama_get_logits_ith(ctx, -1);
                second_row.assign(logits, logits + vocab);
            }
            if (step + 1 < steps) {
                batch.clear();
                batch.add(forced[step], prefix_len + step, 0, true);
                if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()))
                    throw std::runtime_error("diagnostic causal step failed");
            }
        }
        restore();
        for (int b = 0; b < blocks; ++b) {
            const auto & tokens = block_ids[b];
            batch.clear();
            for (int i = 0; i < int(tokens.size()); ++i) batch.add(tokens[i], prefix_len + i, 0, true);
            llama_set_causal_attn(ctx, false);
            const int code = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get());
            llama_set_causal_attn(ctx, true);
            if (code) throw std::runtime_error("diagnostic preview failed");
            auto file = output_file("case-" + std::to_string(c) + "-preview-" + std::to_string(b) + ".f32");
            for (int row = 0; row < int(tokens.size()); ++row) write_row(file, row);
            restore();
        }
        batch.clear();
        batch.add(forced[0], prefix_len, 0, true);
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()))
            throw std::runtime_error("diagnostic restoration probe failed");
        const bool equal = std::memcmp(second_row.data(), llama_get_logits_ith(ctx, -1), row_bytes) == 0;
        std::printf("{\"type\":\"clover_diagnostic\",\"case\":%d,\"vocab\":%d,\"steps\":%d,\"blocks\":%d,\"restored_causal_equal\":%s}\n",
            c, vocab, steps, blocks, equal ? "true" : "false");
        std::fflush(stdout);
        if (!equal) throw std::runtime_error("preview changed restored causal logits");
    }
}

int main(int argc, char ** argv) {
    std::string cases_path, output_path;
    std::vector<char *> arguments{argv[0]};
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--clover-cases" || arg == "--clover-output") {
            if (++i == argc) throw std::runtime_error("missing diagnostic argument value");
            (arg == "--clover-cases" ? cases_path : output_path) = argv[i];
        } else {
            arguments.push_back(argv[i]);
        }
    }
    if (cases_path.empty() != output_path.empty()) throw std::runtime_error("diagnostics require both input and output paths");
    arguments.push_back(nullptr);
    common_params params;
    common_init();
    if (!common_params_parse(int(arguments.size()) - 1, arguments.data(), params, LLAMA_EXAMPLE_SPECULATIVE)) return 1;
    const auto limits = common_speculative_get_output_limits(
        params.n_batch, params.n_parallel, common_speculative_n_max(&params.speculative));
    params.n_outputs_max = limits.total;
    params.n_outputs_max_per_seq = limits.per_seq;
    if (!cases_path.empty()) {
        params.n_outputs_max = std::max(params.n_outputs_max, 32);
        params.n_outputs_max_per_seq = std::max(params.n_outputs_max_per_seq, 32);
    }
    llama_backend_init();
    llama_numa_init(params.numa);
    auto initialized = common_init_from_params(params);
    auto * ctx = initialized->context();
    if (!ctx) throw std::runtime_error("stock context initialization failed");
    if (!cases_path.empty()) {
        dump_cases(ctx, cases_path, output_path);
        return 0;
    }
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
