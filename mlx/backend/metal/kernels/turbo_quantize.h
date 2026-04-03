// Copyright © 2024 Apple Inc.

#pragma once

#include <metal_stdlib>
using namespace metal;

///  Fused TurboQuant quantize kernel.
///
///  Per-vector pipeline (one threadgroup per vector):
///    1. Compute L2 norm
///    2. Normalize
///    3. Multiply by random signs
///    4. Walsh-Hadamard Transform (butterfly in shared memory)
///    5. Nearest-centroid lookup via boundary comparison
///    6. Norm correction:  adjusted_norm = orig_norm / ||codebook[indices]||
///    7. Bit-pack indices into uint32 words
///
///  Template parameters:
///    T      — input element type (float16_t / bfloat16_t)
///    D      — head dimension (must be power of 2, e.g. 64, 128)
///    BITS   — quantization bits (3 or 4)
///    N_LVLS — number of codebook levels = 2^BITS (8 or 16)
///    VPW    — values packed per uint32 word (10 for 3-bit, 8 for 4-bit)

template <typename T, int D, int BITS, int N_LVLS, int VPW>
[[kernel]] void turbo_quantize(
    const device T* input [[buffer(0)]],          // [total_vectors, D]
    const device float* signs [[buffer(1)]],      // [D]
    const device float* codebook [[buffer(2)]],   // [N_LVLS]
    device uint32_t* packed_out [[buffer(3)]],    // [total_vectors, PACKED_DIM]
    device float* norms_out [[buffer(4)]],        // [total_vectors]
    const constant int& total_vectors [[buffer(5)]],
    uint tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]]) {

  constexpr int PACKED_DIM = (D + VPW - 1) / VPW;
  constexpr float WHT_SCALE = 1.0f / sqrt(float(D));

  int vec_id = tid;
  if (vec_id >= total_vectors) return;

  // ── Shared memory ────────────────────────────────────────────────────────
  threadgroup float s_vec[D];
  threadgroup float s_codebook[N_LVLS];
  threadgroup float s_boundaries[N_LVLS];  // only [0..N_LVLS-2] used
  threadgroup uint32_t s_shifted[D];

  // Load codebook into shared memory
  if ((int)lid < N_LVLS) {
    s_codebook[lid] = codebook[lid];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Compute midpoint boundaries from codebook
  if ((int)lid < N_LVLS - 1) {
    s_boundaries[lid] = (s_codebook[lid] + s_codebook[lid + 1]) * 0.5f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // ── Step 1: Load input & compute L2 norm ─────────────────────────────────
  float val = float(input[vec_id * D + lid]);
  s_vec[lid] = val * val;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Tree reduction for sum of squares
  for (int stride = D / 2; stride > 0; stride >>= 1) {
    if ((int)lid < stride) {
      s_vec[lid] += s_vec[lid + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float orig_norm = sqrt(s_vec[0]);
  float safe_norm = max(orig_norm, 1e-10f);

  // ── Step 2: Normalize ────────────────────────────────────────────────────
  float normalized = val / safe_norm;

  // ── Step 3: Apply random signs ───────────────────────────────────────────
  float signed_val = normalized * signs[lid];

  // ── Step 4: Walsh-Hadamard Transform (butterfly) ─────────────────────────
  s_vec[lid] = signed_val;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (int half = 1; half < D; half <<= 1) {
    uint partner = lid ^ half;
    float my_val = s_vec[lid];
    float other_val = s_vec[partner];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (!(lid & half)) {
      s_vec[lid] = my_val + other_val;   // lower index: a + b
    } else {
      s_vec[lid] = other_val - my_val;   // upper index: a - b
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  float rotated = s_vec[lid] * WHT_SCALE;

  // ── Step 5: Nearest centroid (linear scan over boundaries) ───────────────
  int index = 0;
  for (int i = 0; i < N_LVLS - 1; i++) {
    if (rotated > s_boundaries[i]) index++;
  }

  // ── Step 6: Norm correction ──────────────────────────────────────────────
  // Compute ||codebook[indices]||² via parallel reduction
  float recon_val = s_codebook[index];
  s_vec[lid] = recon_val * recon_val;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (int stride = D / 2; stride > 0; stride >>= 1) {
    if ((int)lid < stride) {
      s_vec[lid] += s_vec[lid + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float recon_norm = sqrt(max(s_vec[0], 1e-20f));
  float adjusted_norm = orig_norm / recon_norm;

  // Write adjusted norm (one thread per vector)
  if (lid == 0) {
    norms_out[vec_id] = adjusted_norm;
  }

  // ── Step 7: Bit-pack indices ─────────────────────────────────────────────
  int word_idx = (int)lid / VPW;
  int pos_in_word = (int)lid % VPW;
  s_shifted[lid] = (uint32_t(index) << (pos_in_word * BITS));
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // First thread of each word group ORs values together
  if ((int)lid % VPW == 0 && word_idx < PACKED_DIM) {
    uint32_t word = 0;
    int start = (int)lid;
    int end = min(start + VPW, D);
    for (int i = start; i < end; i++) {
      word |= s_shifted[i];
    }
    packed_out[vec_id * PACKED_DIM + word_idx] = word;
  }
}
