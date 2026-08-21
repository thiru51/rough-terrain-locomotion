cmake_minimum_required(VERSION 3.18)
project(tert_inference LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# The context buffer is deliberately free of LibTorch so its indexing can be
# built and tested on a machine with no model, no CUDA and no torch install.
add_library(tert_context INTERFACE)
target_include_directories(tert_context INTERFACE include)

add_executable(test_context_buffer tests/test_context_buffer.cpp)
target_link_libraries(test_context_buffer PRIVATE tert_context)

enable_testing()
add_test(NAME context_buffer COMMAND test_context_buffer)

# LibTorch ships inside the pip torch package, so pointing CMake at the venv is
# usually enough:
#   cmake -DCMAKE_PREFIX_PATH=$(python -c "import torch;print(torch.utils.cmake_prefix_path)") ..
find_package(Torch QUIET)

if(Torch_FOUND)
  add_library(tert_inference src/policy_runner.cpp)
  target_include_directories(tert_inference PUBLIC include)
  target_link_libraries(tert_inference PUBLIC tert_context "${TORCH_LIBRARIES}")

  add_executable(benchmark_policy tools/benchmark_policy.cpp)
  target_link_libraries(benchmark_policy PRIVATE tert_inference)
else()
  message(STATUS "LibTorch not found - building the context buffer and its tests only")
endif()
