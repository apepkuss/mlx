// Copyright © 2026 Apple Inc.

#include <vector>

#include "doctest/doctest.h"

#include "mlx/allocator.h"
#include "mlx/backend/metal/device.h"
#include "mlx/memory.h"
#include "mlx/mlx.h"

using namespace mlx::core;

namespace {

constexpr size_t MB = 1 << 20;

metal::ResidencySets& residency() {
  return metal::device(Device::gpu).residency_sets();
}

// Restores the wired limit, the cache limit and the set cap on scope exit so
// a failing assertion can't leak global state into the rest of the suite. A
// cache limit of 0 makes free() release (and unwire) immediately instead of
// recycling, so residency accounting is observable synchronously.
struct LimitGuard {
  size_t wired;
  size_t cache;
  size_t max_per_set;
  LimitGuard(size_t new_wired, size_t new_cache)
      : cache(set_cache_limit(new_cache)),
        max_per_set(residency().max_bytes_per_set()) {
    clear_cache();
    synchronize(); // retire command buffers still holding temporaries
    wired = set_wired_limit(new_wired);
  }
  ~LimitGuard() {
    residency().set_max_bytes_per_set(max_per_set);
    set_cache_limit(cache);
    set_wired_limit(wired);
  }
};

// Sum of a small graph, used to drive a command-buffer commit (and with it the
// residency attach path) while sets exist. Synchronizes so the command
// buffer's temporaries are released before the caller checks accounting.
void check_gpu_work() {
  auto x = sum(ones({256, 256}, float32));
  eval(x);
  CHECK_EQ(x.item<float>(), 65536.0f);
  synchronize();
}

} // namespace

TEST_CASE("test residency set wires nothing when the wired limit is zero") {
  if (!residency().enabled()) {
    INFO("skipped: needs a Metal 3 GPU on macOS >= 15");
    return;
  }
  LimitGuard guard(0, 0);

  // The default wired limit is 0, and at 0 nothing may be wired -- not even
  // the allocator's heap.
  CHECK_EQ(residency().wired_size(), 0);

  std::vector<allocator::Buffer> bufs;
  for (int i = 0; i < 4; ++i) {
    bufs.push_back(allocator::malloc(4 * MB));
    CHECK_EQ(residency().wired_size(), 0);
  }
  for (auto buf : bufs) {
    allocator::free(buf);
  }
  CHECK_EQ(residency().wired_size(), 0);
}

TEST_CASE("test residency set never wires more than the wired limit") {
  if (!residency().enabled()) {
    return;
  }
  LimitGuard guard(0, 0);
  // Sample the baseline with a budget in place, so the allocator's heap is
  // already wired and only our own allocations move the total.
  set_wired_limit(256 * MB);
  const size_t baseline = residency().wired_size();
  const size_t limit = baseline + 8 * MB;
  set_wired_limit(limit);

  std::vector<allocator::Buffer> bufs;
  for (int i = 0; i < 8; ++i) { // asks for 32 MB against an 8 MB budget
    bufs.push_back(allocator::malloc(4 * MB));
    CHECK_LE(residency().wired_size(), limit);
  }
  // Over budget overall, but the budget itself is still used.
  CHECK_GE(residency().wired_size(), baseline + 4 * MB);

  for (auto buf : bufs) {
    allocator::free(buf);
  }
  // Everything of ours is unwired again, exactly.
  CHECK_EQ(residency().wired_size(), baseline);
}

TEST_CASE("test raising the wired limit wires already-allocated buffers") {
  if (!residency().enabled()) {
    return;
  }
  LimitGuard guard(0, 0);

  auto buf = allocator::malloc(8 * MB);
  CHECK_EQ(residency().wired_size(), 0);

  set_wired_limit(64 * MB);
  CHECK_GE(residency().wired_size(), 8 * MB);

  allocator::free(buf);
}

TEST_CASE("test lowering the wired limit unwires buffers") {
  if (!residency().enabled()) {
    return;
  }
  LimitGuard guard(64 * MB, 0);
  const size_t baseline_with_budget = residency().wired_size();

  auto buf = allocator::malloc(8 * MB);
  CHECK_GE(residency().wired_size(), baseline_with_budget + 8 * MB);

  set_wired_limit(0);
  CHECK_EQ(residency().wired_size(), 0);

  // ...and comes back when the budget is restored.
  set_wired_limit(64 * MB);
  CHECK_GE(residency().wired_size(), baseline_with_budget + 8 * MB);

  allocator::free(buf);
}

TEST_CASE("test residency accounting is exact across free/realloc cycles") {
  if (!residency().enabled()) {
    return;
  }
  LimitGuard guard(64 * MB, 0);

  // Baseline is whatever is wired with nothing of ours allocated (the heap).
  const size_t baseline = residency().wired_size();

  for (int i = 0; i < 32; ++i) {
    auto buf = allocator::malloc(8 * MB);
    CHECK_GE(residency().wired_size(), baseline + 8 * MB);
    allocator::free(buf);
    // Exact: erase subtracts the bytes recorded at insert, so repeated
    // cycles must not drift the running total.
    CHECK_EQ(residency().wired_size(), baseline);
  }
}

TEST_CASE("test wired allocations are spread across size-capped sets") {
  if (!residency().enabled()) {
    return;
  }
  LimitGuard guard(0, 0);
  const size_t max_per_set = 8 * MB;
  residency().set_max_bytes_per_set(max_per_set);
  set_wired_limit(128 * MB);
  const size_t baseline = residency().wired_size();
  const size_t baseline_sets = residency().num_sets();

  // Each buffer is over half the cap, so no two share a set.
  std::vector<allocator::Buffer> bufs;
  for (int i = 0; i < 8; ++i) {
    bufs.push_back(allocator::malloc(5 * MB));
  }
  CHECK_EQ(residency().wired_size(), baseline + 8 * 5 * MB);
  CHECK_GT(residency().num_sets(), baseline_sets);
  // No set may exceed the cap while there is room to make more.
  CHECK_GE(residency().num_sets(), 8);

  // Sets created after the queue exists must still be attached before the
  // commit that could reference them.
  check_gpu_work();

  const size_t sets_before = residency().num_sets();
  for (auto buf : bufs) {
    allocator::free(buf);
  }
  CHECK_EQ(residency().wired_size(), baseline);

  // Emptied sets are reused rather than leaked, so cycling the same
  // allocations must not keep growing the set count.
  for (int cycle = 0; cycle < 4; ++cycle) {
    std::vector<allocator::Buffer> again;
    for (int i = 0; i < 8; ++i) {
      again.push_back(allocator::malloc(5 * MB));
    }
    CHECK_EQ(residency().num_sets(), sets_before);
    for (auto buf : again) {
      allocator::free(buf);
    }
  }
  CHECK_EQ(residency().wired_size(), baseline);
}

TEST_CASE("test the set count stays within the command queue limit") {
  if (!residency().enabled()) {
    return;
  }
  LimitGuard guard(0, 0);
  // A tiny cap against a large budget asks for far more sets than a command
  // queue accepts; the count must saturate instead.
  residency().set_max_bytes_per_set(2 * MB);
  set_wired_limit(256 * MB);
  const size_t baseline = residency().wired_size();

  std::vector<allocator::Buffer> bufs;
  for (int i = 0; i < 64; ++i) {
    bufs.push_back(allocator::malloc(2 * MB));
  }
  CHECK_LE(residency().num_sets(), 32);
  CHECK_EQ(residency().wired_size(), baseline + 64 * 2 * MB);

  // Attaching a saturated set set to a queue must not fail.
  check_gpu_work();

  for (auto buf : bufs) {
    allocator::free(buf);
  }
  CHECK_EQ(residency().wired_size(), baseline);
}

TEST_CASE("test an allocation larger than the set cap is still wired") {
  if (!residency().enabled()) {
    return;
  }
  LimitGuard guard(0, 0);
  residency().set_max_bytes_per_set(4 * MB);
  set_wired_limit(128 * MB);
  const size_t baseline = residency().wired_size();

  auto big = allocator::malloc(32 * MB);
  CHECK_EQ(residency().wired_size(), baseline + 32 * MB);
  check_gpu_work();

  allocator::free(big);
  CHECK_EQ(residency().wired_size(), baseline);
}

TEST_CASE("test a set cap of zero keeps everything in one set") {
  if (!residency().enabled()) {
    return;
  }
  LimitGuard guard(0, 0);
  residency().set_max_bytes_per_set(0);
  set_wired_limit(128 * MB);
  const size_t baseline = residency().wired_size();
  const size_t baseline_sets = residency().num_sets();

  std::vector<allocator::Buffer> bufs;
  for (int i = 0; i < 16; ++i) {
    bufs.push_back(allocator::malloc(4 * MB));
  }
  CHECK_EQ(residency().num_sets(), baseline_sets);
  CHECK_EQ(residency().wired_size(), baseline + 16 * 4 * MB);

  for (auto buf : bufs) {
    allocator::free(buf);
  }
  CHECK_EQ(residency().wired_size(), baseline);
}

TEST_CASE("test the wired limit is used up to its boundary") {
  if (!residency().enabled()) {
    return;
  }
  LimitGuard guard(0, 0);
  // Sample the baseline with a budget in place, then leave room for exactly one
  // more 4 MB allocation.
  set_wired_limit(256 * MB);
  const size_t baseline = residency().wired_size();
  set_wired_limit(baseline + 4 * MB);

  auto first = allocator::malloc(4 * MB);
  CHECK_EQ(residency().wired_size(), baseline + 4 * MB);

  // The budget is exactly full: the next allocation is tracked but not wired.
  auto second = allocator::malloc(4 * MB);
  CHECK_EQ(residency().wired_size(), baseline + 4 * MB);

  // Freeing the wired one does not promote the pending one.
  allocator::free(first);
  CHECK_EQ(residency().wired_size(), baseline);

  allocator::free(second);
  CHECK_EQ(residency().wired_size(), baseline);
}

TEST_CASE("test a pinned buffer is resident outside the wired limit") {
  if (!residency().enabled()) {
    return;
  }
  LimitGuard guard(0, 0);
  CHECK_EQ(residency().pinned_size(), 0);
  CHECK_FALSE(residency().has_pinned());

  auto a = allocator::malloc(4 * MB);
  auto b = allocator::malloc(4 * MB);
  auto mtl = [](allocator::Buffer buf) {
    return static_cast<MTL::Buffer*>(buf.ptr());
  };
  // Nothing is wired at a limit of zero, yet the pinned one is resident.
  CHECK_EQ(residency().pin(mtl(a)), 4 * MB);
  CHECK_EQ(residency().pin(mtl(a)), 0); // already pinned
  CHECK_EQ(residency().pinned_size(), 4 * MB);
  CHECK_EQ(residency().wired_size(), 0);
  CHECK(residency().has_pinned());
  check_gpu_work();

  // Raising and lowering the limit moves only what is not pinned.
  set_wired_limit(256 * MB);
  const size_t wired = residency().wired_size();
  CHECK_GE(wired, 4 * MB); // b
  CHECK_EQ(residency().pinned_size(), 4 * MB);
  set_wired_limit(0);
  CHECK_EQ(residency().wired_size(), 0);
  CHECK_EQ(residency().pinned_size(), 4 * MB);

  // Unpinned with no budget left, it is no longer resident.
  CHECK_EQ(residency().unpin(mtl(a)), 4 * MB);
  CHECK_EQ(residency().unpin(mtl(a)), 0);
  CHECK_EQ(residency().pinned_size(), 0);
  CHECK_EQ(residency().wired_size(), 0);

  // Unpinned while the limit has room, it stays resident under the limit.
  set_wired_limit(256 * MB);
  const size_t before = residency().wired_size();
  CHECK_EQ(residency().pin(mtl(b)), 4 * MB);
  CHECK_EQ(residency().wired_size(), before - 4 * MB);
  CHECK_EQ(residency().unpin(mtl(b)), 4 * MB);
  CHECK_EQ(residency().wired_size(), before);

  allocator::free(a);
  allocator::free(b);
}

TEST_CASE("test freeing a pinned buffer ends the pin") {
  if (!residency().enabled()) {
    return;
  }
  auto mtl = [](allocator::Buffer buf) {
    return static_cast<MTL::Buffer*>(buf.ptr());
  };
  {
    // Released on free.
    LimitGuard guard(0, 0);
    auto a = allocator::malloc(4 * MB);
    CHECK_EQ(residency().pin(mtl(a)), 4 * MB);
    allocator::free(a);
    CHECK_EQ(residency().pinned_size(), 0);
    CHECK_FALSE(residency().has_pinned());
    CHECK_EQ(residency().wired_size(), 0);
  }
  {
    // Kept in the buffer cache on free: whatever reuses it is not pinned.
    LimitGuard guard(0, 64 * MB);
    auto a = allocator::malloc(4 * MB);
    void* ptr = a.ptr();
    CHECK_EQ(residency().pin(mtl(a)), 4 * MB);
    allocator::free(a);
    CHECK_EQ(residency().pinned_size(), 0);
    auto b = allocator::malloc(4 * MB);
    CHECK_EQ(b.ptr(), ptr); // the cached buffer
    CHECK_EQ(residency().pinned_size(), 0);
    CHECK_EQ(residency().wired_size(), 0);
    allocator::free(b);
    clear_cache();
  }
}

TEST_CASE("test pin_resident pins the buffers of evaluated arrays") {
  if (!residency().enabled()) {
    return;
  }
  LimitGuard guard(0, 0);
  {
    auto x = ones({2 * 1024, 1024}, float32); // 8 MB
    auto lazy = x + 1;
    eval(x);
    synchronize();
    // An array that is not evaluated has no buffer to pin.
    CHECK_EQ(pin_resident({lazy}), 0);
    const size_t pinned = pin_resident({x, x});
    CHECK_GE(pinned, 8 * MB);
    CHECK_EQ(get_pinned_memory(), pinned);
    CHECK_EQ(pin_resident({x}), 0);
    CHECK_EQ(unpin_resident({x}), pinned);
    CHECK_EQ(get_pinned_memory(), 0);
    CHECK_EQ(pin_resident({x}), pinned);
  }
  // The array is gone, and its pin with it.
  synchronize();
  CHECK_EQ(get_pinned_memory(), 0);
}
