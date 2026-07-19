// Copyright © 2026 MLX Contributors

#include "doctest/doctest.h"

#include <atomic>
#include <exception>
#include <string>
#include <thread>
#include <vector>

#include "mlx/backend/metal/device.h"
#include "mlx/device.h"

using namespace mlx::core;

TEST_CASE("test concurrent first kernel lookup and library clearing") {
  if (!is_available(Device::gpu)) {
    return;
  }

  constexpr int num_threads = 8;
  constexpr const char* kernel_name = "kernel_cache_first_lookup";
  constexpr const char* source = R"metal(
#include <metal_stdlib>
using namespace metal;

kernel void kernel_cache_first_lookup(
    device float* out [[buffer(0)]],
    uint elem [[thread_position_in_grid]]) {
  out[elem] = float(elem);
}
)metal";

  auto& device = metal::device(Device::gpu);
  std::vector<std::string> library_names;
  std::vector<MTL::Library*> libraries;
  std::vector<std::string> clear_library_names;
  library_names.reserve(num_threads);
  libraries.reserve(num_threads);
  clear_library_names.reserve(num_threads);
  for (int i = 0; i < num_threads; ++i) {
    library_names.push_back(
        "concurrent_kernel_cache_first_lookup_" + std::to_string(i));
    libraries.push_back(
        device.get_library(library_names.back(), [source] { return source; }));

    clear_library_names.push_back(
        "concurrent_kernel_cache_clear_" + std::to_string(i));
    auto* clear_library = device.get_library(
        clear_library_names.back(), [source] { return source; });
    REQUIRE(device.get_kernel(kernel_name, clear_library) != nullptr);
  }

  std::atomic<int> ready{0};
  std::atomic<bool> start{false};
  std::vector<MTL::ComputePipelineState*> kernels(num_threads, nullptr);
  std::vector<std::exception_ptr> errors(num_threads);
  std::vector<std::thread> threads;
  threads.reserve(2 * num_threads);
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back([&, i] {
      ready.fetch_add(1, std::memory_order_release);
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      try {
        kernels[i] = device.get_kernel(kernel_name, libraries[i]);
      } catch (...) {
        errors[i] = std::current_exception();
      }
    });
    threads.emplace_back([&, i] {
      ready.fetch_add(1, std::memory_order_release);
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      device.clear_library(clear_library_names[i]);
    });
  }
  while (ready.load(std::memory_order_acquire) != 2 * num_threads) {
    std::this_thread::yield();
  }
  start.store(true, std::memory_order_release);
  for (auto& thread : threads) {
    thread.join();
  }

  for (int i = 0; i < num_threads; ++i) {
    CHECK(errors[i] == nullptr);
    CHECK(kernels[i] != nullptr);
    device.clear_library(library_names[i]);
  }
}
