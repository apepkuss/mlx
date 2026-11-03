#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "mlx/mlx.h"
#include "mlx/scheduler.h"

using namespace mlx::core;

int identity() {
  bool worker_is_main = true;
  std::thread([&] { worker_is_main = is_main_thread(); }).join();
  if (worker_is_main || !is_main_thread()) {
    return 1;
  }
  for (int i = 0; i < 256; ++i) {
    std::thread([&] { worker_is_main = is_main_thread(); }).join();
    if (worker_is_main) {
      return 2;
    }
  }
  return 0;
}

int graph(Device device) {
  std::vector<array> outputs;
  std::exception_ptr failure;
  std::thread worker([&] {
    try {
      auto stream = new_thread_unsafe_stream(device);
      auto x = ones({4, 5}, float32, stream);
      eval(x);
      outputs.push_back(add(x, array(2.0f), stream));
      auto pending = add(ones({16}, float32, stream), array(3.0f), stream);
      async_eval(pending);
      outputs.push_back(pending);
      clear_streams();
    } catch (...) {
      failure = std::current_exception();
    }
  });
  worker.join();
  if (failure) {
    std::rethrow_exception(failure);
  }
  eval(outputs);
  return sum(outputs[0], false, device).item<float>() == 60.0f &&
          sum(outputs[1], false, device).item<float>() == 64.0f
      ? 0
      : 3;
}

int pending(Device device) {
  std::vector<array> outputs;
  for (int i = 0; i < 64; ++i) {
    std::exception_ptr failure;
    std::thread worker([&] {
      try {
        auto output = multiply(ones({1024}, float32, device), array(3.0f), device);
        async_eval(output);
        outputs.push_back(output);
      } catch (...) {
        failure = std::current_exception();
      }
    });
    worker.join();
    if (failure) {
      std::rethrow_exception(failure);
    }
  }
  eval(outputs);
  for (auto& output : outputs) {
    if (sum(output, false, device).item<float>() != 3072.0f) {
      return 4;
    }
  }
  clear_streams();
  return 0;
}

int worker_error() {
  bool caught = false;
  std::thread worker([&] {
    auto stream = default_stream(Device::cpu);
    scheduler::enqueue(stream, [] {
      throw std::runtime_error("worker error propagation");
    });
    try {
      synchronize(stream);
    } catch (const std::exception& error) {
      caught = std::string(error.what()).find("worker error propagation") !=
          std::string::npos;
    }
  });
  worker.join();
  return caught ? 0 : 6;
}

int main(int argc, char** argv) {
  try {
    auto mode = argc > 1 ? std::string(argv[1]) : "identity";
    if (mode == "identity") {
      return identity();
    }
    if (mode == "error") {
      return worker_error();
    }
    auto device = argc > 2 && std::string(argv[2]) == "cpu" ? Device::cpu
                                                            : Device::gpu;
    return mode == "graph" ? graph(device) : pending(device);
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 5;
  }
}
