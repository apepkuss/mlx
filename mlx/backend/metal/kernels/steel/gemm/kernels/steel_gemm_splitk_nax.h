// Copyright © 2026 Apple Inc.

using namespace mlx::steel;

constant bool align_M [[function_constant(200)]];
constant bool align_N [[function_constant(201)]];

///////////////////////////////////////////////////////////////////////////////
// NAX Split-K GEMM kernel
///////////////////////////////////////////////////////////////////////////////

// clang-format off
template <
    typename T,
    int BM,
    int BN,
    int BK,
    int WM,
    int WN,
    bool transpose_a,
    bool transpose_b,
    typename AccumType = float>
[[kernel, max_total_threads_per_threadgroup(WM* WN * 32)]] void gemm_splitk_nax(
    const device T* A [[buffer(0)]],
    const device T* B [[buffer(1)]],
    device AccumType* C [[buffer(2)]],
    const constant GEMMSpiltKParams* params [[buffer(3)]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]]) { // clang-format on

  const int linear_tid = tid.x;

  // Compute swizzled tile dimensions
  const int tn_swizzled = params->tiles_n << params->swizzle_log;
  const int tm_swizzled =
      (params->tiles_m + (1 << params->swizzle_log) - 1) >> params->swizzle_log;
  const int tiles_per_partition = tn_swizzled * tm_swizzled;

  const int tid_z = linear_tid / tiles_per_partition;
  const int xy_flat = linear_tid % tiles_per_partition;

  // Decode 2D grid coordinates in swizzled space
  const int grid_x = xy_flat % tn_swizzled;
  const int grid_y = xy_flat / tn_swizzled;

  // Apply X-Y swizzle
  const int tid_y = (grid_y << params->swizzle_log) +
      (grid_x & ((1 << params->swizzle_log) - 1));
  const int tid_x = grid_x >> params->swizzle_log;

  // Exit early
  if (params->tiles_n <= tid_x || params->tiles_m <= tid_y) {
    return;
  }

  // Calculate partition bounds
  const int c_row = tid_y * BM;
  const int c_col = tid_x * BN;
  const int k_start = params->split_k_partition_size * tid_z;
  const int k_end = min(k_start + params->split_k_partition_size, params->K);

  const size_t c_row_long = size_t(c_row);
  const size_t c_col_long = size_t(c_col);
  const size_t k_start_long = size_t(k_start);

  // Adjust pointers for split-K partition
  A += transpose_a ? (c_row_long + k_start_long * params->lda)
                   : (k_start_long + c_row_long * params->lda);
  B += transpose_b ? (k_start_long + c_col_long * params->ldb)
                   : (c_col_long + k_start_long * params->ldb);
  C += (size_t(params->split_k_partition_stride) * tid_z) +
      (c_row_long * params->ldc + c_col_long);

  // NAX tile configuration
  constexpr short SM = BM / WM;
  constexpr short SN = BN / WN;

  // Calculate simdgroup offsets and alignment
  const short tm = SM * (simd_group_id / WN);
  const short tn = SN * (simd_group_id % WN);

  const int sgp_sm_int =
      align_M ? int(SM) : max(0, min(int(SM), params->M - (c_row + tm)));
  const short sgp_sm = short(sgp_sm_int);

  const int sgp_sn_int =
      align_N ? int(SN) : max(0, min(int(SN), params->N - (c_col + tn)));
  const short sgp_sn = short(sgp_sn_int);

  A += transpose_a ? tm : (tm * params->lda);
  B += transpose_b ? (tn * params->ldb) : tn;
  C += tm * params->ldc + tn;

  const int partition_k_size = k_end - k_start;

  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
      SM,
      SN,
      static_cast<int>(dynamic_extent),
      transpose_a,
      transpose_b,
      true,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply);

  mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> gemm_op;

  array<int, 2> a_strides = {1, int(params->lda)};
  array<int, 2> b_strides = {1, int(params->ldb)};
  const int a_extent_x = transpose_a ? max(1, int(sgp_sm)) : partition_k_size;
  const int a_extent_y = transpose_a ? partition_k_size : max(1, int(sgp_sm));
  const int b_extent_x = transpose_b ? partition_k_size : max(1, int(sgp_sn));
  const int b_extent_y = transpose_b ? max(1, int(sgp_sn)) : partition_k_size;

  tensor<device T, dextents<int, 2>, tensor_inline> Atensor(
      const_cast<device T*>(A),
      dextents<int, 2>{a_extent_x, a_extent_y},
      a_strides);
  tensor<device T, dextents<int, 2>, tensor_inline> Btensor(
      const_cast<device T*>(B),
      dextents<int, 2>{b_extent_x, b_extent_y},
      b_strides);

  auto Dtensor = gemm_op.template get_destination_cooperative_tensor<
      decltype(Atensor),
      decltype(Btensor),
      AccumType>();

  const bool active = sgp_sm > 0 && sgp_sn > 0 && partition_k_size > 0;
  if (active) {
    gemm_op.run(Atensor, Btensor, Dtensor);
  }

  STEEL_PRAGMA_UNROLL
  for (uint16_t i = 0; i < Dtensor.get_capacity(); ++i) {
    if (Dtensor.is_valid_element(i)) {
      auto coord = Dtensor.get_multidimensional_index(i);
      const int col = coord[0];
      const int row = coord[1];
      if (row < int(sgp_sm) && col < int(sgp_sn)) {
        C[row * params->ldc + col] = Dtensor[i];
      }
    }
  }
}
