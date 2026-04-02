// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <optional>
#include <variant>

#include "mlx/api.h"
#include "mlx/utils.h"

namespace mlx::core::fast {

MLX_API array rms_norm(
    const array& x,
    const std::optional<array>& weight,
    float eps,
    StreamOrDevice s = {});

MLX_API array layer_norm(
    const array& x,
    const std::optional<array>& weight,
    const std::optional<array>& bias,
    float eps,
    StreamOrDevice s = {});

MLX_API array rope(
    const array& x,
    int dims,
    bool traditional,
    std::optional<float> base,
    float scale,
    int offset,
    const std::optional<array>& freqs = std::nullopt,
    StreamOrDevice s = {});

MLX_API array rope(
    const array& x,
    int dims,
    bool traditional,
    std::optional<float> base,
    float scale,
    const array& offset,
    const std::optional<array>& freqs = std::nullopt,
    StreamOrDevice s = {});

/** Computes: O = softmax(Q @ K.T) @ V **/
MLX_API array scaled_dot_product_attention(
    const array& queries,
    const array& keys,
    const array& values,
    const float scale,
    const std::string& mask_mode = "",
    std::optional<array> mask_arr = {},
    const std::optional<array>& sinks = {},
    StreamOrDevice s = {});

/** TurboQuant SDPA: attention with bit-packed KV cache.
 *  K is stored as packed uint32 indices with per-vector norms.
 *  Queries must be pre-rotated: Q_rot = WHT(signs * Q).
 *  V is passed as dequantized fp16. **/
/** TurboQuant SDPA: attention with bit-packed K cache and optional V compression.
 *  K is stored as packed uint32 indices with per-vector norms.
 *  V can be fp16 (v_bits=0) or bit-packed (v_bits=3,4) with v_norms.
 *  Queries must be pre-rotated: Q_rot = WHT(signs * Q). **/
MLX_API array turboquant_sdpa(
    const array& queries,     // pre-rotated (B, H_q, T_q, D)
    const array& k_packed,    // bit-packed K (B, H_kv, T_kv, k_packed_dim)
    const array& values,      // fp16 V or bit-packed V depending on v_bits
    const array& k_norms,     // per-vector K norms (B, H_kv, T_kv)
    const array& k_codebook,  // K centroids (2^k_bits,)
    const array& v_codebook,  // V centroids (2^v_bits,) — ignored when v_bits=0
    float scale,
    int k_bits = 3,
    int v_bits = 0,           // 0 = fp16, 3 or 4 = compressed
    const std::optional<array>& v_norms = {},  // required when v_bits > 0
    const std::string& mask_mode = "",
    std::optional<array> mask_arr = {},
    StreamOrDevice s = {});

using TemplateArg = std::variant<int, bool, Dtype>;
using ScalarArg = std::variant<bool, int, float>;

using CustomKernelFunction = std::function<std::vector<array>(
    const std::vector<array>&,
    const std::vector<Shape>&,
    const std::vector<Dtype>&,
    std::tuple<int, int, int>,
    std::tuple<int, int, int>,
    std::vector<std::pair<std::string, TemplateArg>>,
    std::optional<float>,
    bool,
    StreamOrDevice)>;

MLX_API CustomKernelFunction metal_kernel(
    const std::string& name,
    const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names,
    const std::string& source,
    const std::string& header = "",
    bool ensure_row_contiguous = true,
    bool atomic_outputs = false);

MLX_API CustomKernelFunction cuda_kernel(
    const std::string& name,
    const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names,
    const std::string& source,
    const std::string& header = "",
    bool ensure_row_contiguous = true,
    int shared_memory = 0);

MLX_API std::vector<array> precompiled_cuda_kernel(
    const std::string& name,
    const std::string& compiled_source,
    const std::vector<array>& inputs,
    const std::vector<Shape>& output_shapes,
    const std::vector<Dtype>& output_dtypes,
    const std::vector<ScalarArg>& scalars,
    std::tuple<int, int, int> grid,
    std::tuple<int, int, int> threadgroup,
    int shared_memory = 0,
    std::optional<float> init_value = std::nullopt,
    bool ensure_row_contiguous = false,
    StreamOrDevice s = {});

} // namespace mlx::core::fast
