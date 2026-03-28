// Copyright © 2023-2024 Apple Inc.
#include <memory>

#include "mlx/backend/gpu/eval.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/primitives.h"
#include "mlx/scheduler.h"

namespace mlx::core::gpu {

void new_stream(Stream stream) {
  if (stream.device == mlx::core::Device::gpu) {
    metal::device(stream.device).new_queue(stream.index);
  }
}

inline void check_error(MTL::CommandBuffer* cbuf) {
  if (cbuf->status() == MTL::CommandBufferStatusError) {
    std::ostringstream msg;
    msg << "[METAL] Command buffer execution failed: "
        << cbuf->error()->localizedDescription()->utf8String();
    throw std::runtime_error(msg.str());
  }
}

// Helper: check buffer status and log if stale, returns true if safe
static bool check_buffer_safe(MTL::CommandBuffer* cb, const char* caller, int stream_idx) {
  if (cb == nullptr) {
    fprintf(stderr, "[MLX_CRASH] %s: null buffer! stream=%d\n", caller, stream_idx);
    return false;
  }
  auto st = cb->status();
  if (st >= MTL::CommandBufferStatusCommitted) {
    fprintf(stderr, "[MLX_CRASH] %s: stale buffer! buf=%p status=%d stream=%d\n",
        caller, (void*)cb, (int)st, stream_idx);
    return false;
  }
  return true;
}

void eval(array& arr) {
  auto pool = metal::new_scoped_memory_pool();
  auto s = arr.primitive().stream();
  auto& d = metal::device(s.device);
  auto command_buffer = d.get_command_buffer(s.index);

  auto outputs = arr.outputs();
  {
    std::vector<array> inputs;
    if (arr.is_tracer()) {
      inputs = arr.inputs();
    }

    debug_set_primitive_buffer_label(command_buffer, arr.primitive());
    arr.primitive().eval_gpu(arr.inputs(), outputs);
  }
  std::unordered_set<std::shared_ptr<array::Data>> buffers;
  for (auto& in : arr.inputs()) {
    buffers.insert(in.data_shared_ptr());
  }
  for (auto& s : arr.siblings()) {
    buffers.insert(s.data_shared_ptr());
  }
  if (auto it = buffers.find(arr.data_shared_ptr()); it != buffers.end()) {
    buffers.erase(it);
  }

  // Re-fetch: eval_gpu() may have triggered cross-stream commit
  command_buffer = d.get_command_buffer(s.index);

  if (d.command_buffer_needs_commit(s.index)) {
    d.end_encoding(s.index);
    scheduler::notify_new_task(s);
    command_buffer = d.get_command_buffer(s.index);
    if (!check_buffer_safe(command_buffer, "eval(commit)", s.index)) {
      d.get_command_buffer(s.index);
      return;
    }
    command_buffer->addCompletedHandler(
        [s, buffers = std::move(buffers)](MTL::CommandBuffer* cbuf) {
          scheduler::notify_task_completion(s);
          check_error(cbuf);
        });
    d.commit_command_buffer(s.index);
    d.get_command_buffer(s.index);
  } else {
    if (!check_buffer_safe(command_buffer, "eval(no-commit)", s.index)) {
      return;
    }
    command_buffer->addCompletedHandler(
        [buffers = std::move(buffers)](MTL::CommandBuffer* cbuf) {
          check_error(cbuf);
        });
  }
}

void finalize(Stream s) {
  auto pool = metal::new_scoped_memory_pool();
  auto& d = metal::device(s.device);
  d.end_encoding(s.index);
  auto cb = d.get_command_buffer(s.index);
  if (!check_buffer_safe(cb, "finalize", s.index)) {
    d.get_command_buffer(s.index);
    return;
  }
  cb->addCompletedHandler([](MTL::CommandBuffer* cbuf) { check_error(cbuf); });
  d.commit_command_buffer(s.index);
  d.get_command_buffer(s.index);
}

void synchronize(Stream s) {
  auto pool = metal::new_scoped_memory_pool();
  auto& d = metal::device(s.device);
  d.end_encoding(s.index);
  auto cb = d.get_command_buffer(s.index);
  if (!check_buffer_safe(cb, "synchronize", s.index)) {
    return;
  }
  cb->retain();
  d.commit_command_buffer(s.index);
  cb->waitUntilCompleted();
  check_error(cb);
  cb->release();
}

} // namespace mlx::core::gpu
