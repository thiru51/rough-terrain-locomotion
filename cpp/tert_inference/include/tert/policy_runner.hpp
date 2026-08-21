// Loads a TorchScript policy and steps it at the control rate.
#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include <torch/script.h>

#include "tert/context_buffer.hpp"

namespace tert {

struct RunnerConfig {
  std::size_t context_len = 20;
  std::size_t obs_dim = 48;
  std::size_t act_dim = 12;
  float action_scale = 0.25f;
  float action_clip = 4.0f;
  bool use_cuda = false;  // A 40-token model is faster on CPU than the transfer costs.
};

struct StepTiming {
  double inference_us = 0.0;
  double total_us = 0.0;
};

class PolicyRunner {
 public:
  PolicyRunner(const std::string& model_path, const RunnerConfig& cfg);

  void reset() { buffer_.reset(); }

  // Observation in, joint position offsets out. The caller adds the default
  // pose and runs PD; this returns the policy's action, already scaled.
  const std::vector<float>& step(const float* observation);

  const StepTiming& timing() const { return timing_; }
  const ContextBuffer& buffer() const { return buffer_; }

 private:
  RunnerConfig cfg_;
  torch::jit::script::Module module_;
  torch::Device device_;
  ContextBuffer buffer_;
  std::vector<float> action_;
  StepTiming timing_;
};

}  // namespace tert
