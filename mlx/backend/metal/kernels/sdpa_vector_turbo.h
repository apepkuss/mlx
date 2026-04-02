// Copyright © 2025 ironmlx contributors
// TurboQuant SDPA vector kernel: decode-time attention with bit-packed KV cache.
//
// K cache: stored as bit-packed uint32 indices + per-vector float32 norms.
//          Queries must be pre-rotated: Q_rot = WHT(signs * Q).
//          Score = dot(Q_rot, codebook[K_indices]) * K_norm
//
// V cache: stored as dequantized fp16 (from incremental decode buffer).
//
// Based on sdpa_vector.h with modified key reading to use codebook dequant.

// NOTE: function_constants and metal includes are provided by the
// parent .metal file that includes this header.

// Template params:
//   T: output type (float16/bfloat16)
//   D: head dimension (64, 128)
//   V_DIM: value dimension (usually == D)
//   K_BITS: K quantization bits (2, 3, 4)
//   K_VPW: K values per uint32 word
//   V_BITS: V quantization bits (0 = fp16 uncompressed, 2, 3, 4)
//   V_VPW: V values per uint32 word (ignored when V_BITS == 0)
template <typename T, int D, int V_DIM = D, int K_BITS = 3, int K_VPW = 10,
          int V_BITS = 0, int V_VPW = 8>
[[kernel]] void sdpa_vector_turbo(
    const device T* queries [[buffer(0)]],
    const device uint32_t* k_packed [[buffer(1)]],
    const device void* v_buf [[buffer(2)]],    // fp16 when V_BITS==0, uint32 packed when V_BITS>0
    device T* out [[buffer(3)]],
    const constant int& gqa_factor [[buffer(4)]],
    const constant int& N [[buffer(5)]],
    const constant size_t& k_head_stride [[buffer(6)]],
    const constant size_t& k_seq_stride [[buffer(7)]],
    const constant size_t& v_head_stride [[buffer(8)]],
    const constant size_t& v_seq_stride [[buffer(9)]],
    const constant float& scale [[buffer(10)]],
    const device bool* bmask [[buffer(11), function_constant(bool_mask)]],
    const device T* fmask [[buffer(12), function_constant(float_mask)]],
    const constant int& mask_kv_seq_stride
    [[buffer(13), function_constant(has_mask)]],
    const constant int& mask_q_seq_stride
    [[buffer(14), function_constant(has_mask)]],
    const constant int& mask_head_stride
    [[buffer(15), function_constant(has_mask)]],
    const device T* sinks [[buffer(16), function_constant(has_sinks)]],
    const constant int& num_q_heads
    [[buffer(17), function_constant(has_sinks)]],
    const device float* k_norms [[buffer(18)]],
    const constant size_t& k_norm_head_stride [[buffer(19)]],
    const device float* k_codebook [[buffer(20)]],
    const device float* v_codebook [[buffer(21)]],
    const device float* v_norms [[buffer(22)]],
    const constant size_t& v_norm_head_stride [[buffer(23)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = V_DIM / BD;
  constexpr uint K_BIT_MASK = (1u << K_BITS) - 1u;
  constexpr uint V_BIT_MASK = V_BITS > 0 ? (1u << V_BITS) - 1u : 0u;

  int inner_v_stride = BN * int(v_seq_stride);

  typedef float U;

  thread U q[qk_per_thread];
  thread U o[v_per_thread];

  threadgroup U outputs[BN * BD];
  threadgroup U max_scores[BN];
  threadgroup U sum_exp_scores[BN];

  // Adjust positions
  const int q_batch_head_idx = tid.x;
  const int q_seq_idx = tid.y;
  const int kv_head_idx = q_batch_head_idx / gqa_factor;
  const int o_offset = q_batch_head_idx * tpg.y + q_seq_idx;
  const int q_offset =
      query_transposed ? tpg.x * q_seq_idx + q_batch_head_idx : o_offset;

  // Query pointer (pre-rotated)
  queries += q_offset * D + simd_lid * qk_per_thread;

  // K packed pointer: navigate to correct head
  k_packed += kv_head_idx * k_head_stride + simd_gid * k_seq_stride;
  k_norms += kv_head_idx * k_norm_head_stride + simd_gid;

  // V pointer: fp16 when V_BITS==0, packed uint32 when V_BITS>0
  const device T* values_fp = (const device T*)v_buf;
  const device uint32_t* v_packed = (const device uint32_t*)v_buf;
  if (V_BITS == 0) {
    values_fp += kv_head_idx * v_head_stride + simd_gid * v_seq_stride +
        simd_lid * v_per_thread;
  } else {
    v_packed += kv_head_idx * v_head_stride + simd_gid * v_seq_stride;
    v_norms += kv_head_idx * v_norm_head_stride + simd_gid;
  }

  if (bool_mask) {
    bmask += q_batch_head_idx * mask_head_stride +
        simd_gid * mask_kv_seq_stride + q_seq_idx * mask_q_seq_stride;
  }
  if (float_mask) {
    fmask += q_batch_head_idx * mask_head_stride +
        simd_gid * mask_kv_seq_stride + q_seq_idx * mask_q_seq_stride;
  }

  out += o_offset * V_DIM + simd_gid * v_per_thread;

  // Read pre-rotated query (with scale)
  for (int i = 0; i < qk_per_thread; i++) {
    q[i] = static_cast<U>(scale) * queries[i];
  }
  for (int i = 0; i < v_per_thread; i++) {
    o[i] = 0;
  }

  U max_score = -1e9f;
  U sum_exp_score = 0;
  if (has_sinks && simd_gid == 0) {
    max_score = static_cast<U>(sinks[q_batch_head_idx % num_q_heads]);
    sum_exp_score = 1;
  }

  // For each key position
  for (int i = simd_gid; i < N; i += BN) {
    bool use_key = true;
    if (do_causal) {
      use_key = i <= (N - int(tpg.y) + int(q_seq_idx));
    } else if (bool_mask) {
      use_key = bmask[0];
    } else if (float_mask) {
      use_key = (fmask[0] >= -1e9f);
    }
    if (use_key) {
      // --- TurboQuant: read bit-packed K indices, codebook lookup ---
      // K indices are bit-packed: VPW values per uint32 word.
      // 3-bit: 10 values per word (30 of 32 bits used).
      // 4-bit: 8 values per word (32 bits used).
      U score = 0;
      int elem_start = simd_lid * qk_per_thread;
      for (int j = 0; j < qk_per_thread; j++) {
        int elem = elem_start + j;
        int word_idx = elem / K_VPW;
        int pos_in_word = elem % K_VPW;
        uint word = k_packed[word_idx];
        uint idx = (word >> (pos_in_word * K_BITS)) & K_BIT_MASK;
        U k_val = k_codebook[idx];
        score += q[j] * k_val;
      }

      // Apply norm: score = dot(q_rot, codebook[indices]) * norm
      U norm_val = k_norms[0];
      score = simd_sum(score) * norm_val;

      if (float_mask) {
        score += static_cast<U>(fmask[0]);
      }

      // Update accumulators (online safe softmax)
      U new_max = max(max_score, score);
      U factor = fast::exp(max_score - new_max);
      U exp_score = fast::exp(score - new_max);

      max_score = new_max;
      sum_exp_score = sum_exp_score * factor + exp_score;

      // Update output with V
      if (V_BITS == 0) {
        // V is fp16: read directly
        for (int j = 0; j < v_per_thread; j++) {
          o[j] = o[j] * factor + exp_score * static_cast<U>(values_fp[j]);
        }
      } else {
        // V is bit-packed: codebook dequant with norm
        U vn = v_norms[0];
        int v_elem_start = simd_lid * v_per_thread;
        for (int j = 0; j < v_per_thread; j++) {
          int ve = v_elem_start + j;
          int vw = ve / V_VPW;
          int vp = ve % V_VPW;
          uint vword = v_packed[vw];
          uint vidx = (vword >> (vp * V_BITS)) & V_BIT_MASK;
          U vval = v_codebook[vidx] * vn;
          o[j] = o[j] * factor + exp_score * vval;
        }
      }
    }

    // Advance pointers
    k_packed += BN * k_seq_stride;
    k_norms += BN;
    if (V_BITS == 0) {
      values_fp += inner_v_stride;
    } else {
      v_packed += BN * v_seq_stride;
      v_norms += BN;
    }
    if (bool_mask) {
      bmask += BN * mask_kv_seq_stride;
    }
    if (float_mask) {
      fmask += BN * mask_kv_seq_stride;
    }
  }

  // Reduction across SIMD groups
  if (simd_lid == 0) {
    max_scores[simd_gid] = max_score;
    sum_exp_scores[simd_gid] = sum_exp_score;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  max_score = max_scores[simd_lid];
  U new_max = simd_max(max_score);
  U factor = fast::exp(max_score - new_max);
  sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);

  for (int i = 0; i < v_per_thread; i++) {
    outputs[simd_lid * BD + simd_gid] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * factor);
    o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Write output
  if (simd_lid == 0) {
    for (int i = 0; i < v_per_thread; i++) {
      out[i] = static_cast<T>(o[i]);
    }
  }
}
