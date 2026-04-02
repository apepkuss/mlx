#include <metal_stdlib>

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/sdpa_vector.h"

using namespace metal;

// SDPA vector instantiations
#define instantiate_sdpa_vector_aggregation(type, value_dim) \
  instantiate_kernel(                                        \
      "sdpa_vector_2pass_2_" #type "_" #value_dim,           \
      sdpa_vector_2pass_2,                                   \
      type,                                                  \
      value_dim)

#define instantiate_sdpa_vector(type, qk_dim, value_dim)       \
  instantiate_kernel(                                          \
      "sdpa_vector_" #type "_" #qk_dim "_" #value_dim,         \
      sdpa_vector,                                             \
      type,                                                    \
      qk_dim,                                                  \
      value_dim)                                               \
  instantiate_kernel(                                          \
      "sdpa_vector_2pass_1_" #type "_" #qk_dim "_" #value_dim, \
      sdpa_vector_2pass_1,                                     \
      type,                                                    \
      qk_dim,                                                  \
      value_dim)

#define instantiate_sdpa_vector_heads(type)      \
  instantiate_sdpa_vector(type, 64, 64)          \
  instantiate_sdpa_vector(type, 96, 96)          \
  instantiate_sdpa_vector(type, 128, 128)        \
  instantiate_sdpa_vector(type, 256, 256)        \
  instantiate_sdpa_vector_aggregation(type, 64)  \
  instantiate_sdpa_vector_aggregation(type, 96)  \
  instantiate_sdpa_vector_aggregation(type, 128) \
  instantiate_sdpa_vector_aggregation(type, 256)

instantiate_sdpa_vector_heads(float)
instantiate_sdpa_vector_heads(bfloat16_t)
instantiate_sdpa_vector_heads(float16_t)

// TurboQuant SDPA: bit-packed K with codebook dequant
#include "mlx/backend/metal/kernels/sdpa_vector_turbo.h"

// Instantiate with K_BITS, K_VPW, V_BITS, V_VPW
#define instantiate_sdpa_vector_turbo(type, dim, kb, kvpw, vb, vvpw) \
  instantiate_kernel(                                                 \
      "sdpa_vector_turbo_" #type "_" #dim                             \
      "_kb" #kb "_kvpw" #kvpw "_vb" #vb "_vvpw" #vvpw,              \
      sdpa_vector_turbo,                                              \
      type,                                                           \
      dim,                                                            \
      dim,                                                            \
      kb,                                                             \
      kvpw,                                                           \
      vb,                                                             \
      vvpw)

#define instantiate_sdpa_vector_turbo_heads(type)                     \
  instantiate_sdpa_vector_turbo(type, 128, 3, 10, 0, 8)              \
  instantiate_sdpa_vector_turbo(type, 128, 3, 10, 4, 8)              \
  instantiate_sdpa_vector_turbo(type, 128, 4, 8, 0, 8)               \
  instantiate_sdpa_vector_turbo(type, 128, 4, 8, 4, 8)               \
  instantiate_sdpa_vector_turbo(type, 64, 3, 10, 0, 8)               \
  instantiate_sdpa_vector_turbo(type, 64, 3, 10, 4, 8)

instantiate_sdpa_vector_turbo_heads(float16_t)
instantiate_sdpa_vector_turbo_heads(bfloat16_t)
    // clang-format on
