// Windows host bindings: keep Python templates out of the nvcc translation unit.
#include <torch/extension.h>

torch::Tensor ggml_dequantize(torch::Tensor, int64_t, int64_t, int64_t,
                             std::optional<at::ScalarType> const&);
torch::Tensor ggml_mul_mat_vec_a8(torch::Tensor, torch::Tensor, int64_t, int64_t);
torch::Tensor ggml_mul_mat_a8(torch::Tensor, torch::Tensor, int64_t, int64_t);
torch::Tensor ggml_moe_a8(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                        torch::Tensor, int64_t, int64_t, int64_t, int64_t);
torch::Tensor ggml_moe_a8_vec(torch::Tensor, torch::Tensor, torch::Tensor,
                            int64_t, int64_t, int64_t, int64_t);
int64_t ggml_moe_get_block_size(int64_t);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ggml_dequantize", &ggml_dequantize, "");
  m.def("ggml_mul_mat_vec_a8", &ggml_mul_mat_vec_a8, "");
  m.def("ggml_mul_mat_a8", &ggml_mul_mat_a8, "");
  m.def("ggml_moe_a8", &ggml_moe_a8, "");
  m.def("ggml_moe_a8_vec", &ggml_moe_a8_vec, "");
  m.def("ggml_moe_get_block_size", &ggml_moe_get_block_size, "");
}
