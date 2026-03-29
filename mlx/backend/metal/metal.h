// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <string>
#include <unordered_map>
#include <variant>

#include "mlx/api.h"

namespace mlx::core::metal {

/* Check if the Metal backend is available. */
MLX_API bool is_available();

/** Capture a GPU trace, saving it to an absolute file `path` */
MLX_API void start_capture(std::string path = "");
MLX_API void stop_capture();

/** Get information about the GPU and system settings. */
MLX_API const
    std::unordered_map<std::string, std::variant<std::string, size_t>>&
    device_info();

/** Get the maximum number of operations per Metal command buffer. */
MLX_API int get_max_ops_per_buffer();

/** Get the maximum MB of data per Metal command buffer. */
MLX_API int get_max_mb_per_buffer();

/** Set the maximum number of operations per Metal command buffer. */
MLX_API void set_max_ops_per_buffer(int val);

/** Set the maximum MB of data per Metal command buffer. */
MLX_API void set_max_mb_per_buffer(int val);

} // namespace mlx::core::metal
