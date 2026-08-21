#include "tert/policy_runner.hpp"

#include <algorithm>
#include <chrono>
#include <stdexcept>

namespace tert {
namespace {

using Clock = std::chrono::steady_clock;

double micros_since(Clock::time_point start) {
  return std::chrono::duration<double, std::micro>(Clock::now() - start).count();
}

}  // namespace

PolicyRunner::PolicyRunner(const std::string& model_path, const RunnerConfig& cfg)
    : cfg_(cfg),
      device_(cfg.use_cuda && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU),
      buffer_(cfg.context_len, cfg.obs_dim, cfg.act_dim),
      action_(cfg.act_dim, 0.0f) {
  try {
    module_ = torch::jit::load(model_path, device_);
  } catch (const c10::Error& e) {
    throw std::runtime_error("failed to load policy from " + model_path + ": " + e.what());
  }
  module_.eval();

  // Two throwaway passes: the first allocates workspaces and, on CUDA, selects
  // kernels. Paying that on the first real control step would blow the deadline.
  torch::NoGradGuard no_grad;
  for (int i = 0; i < 2; ++i) step(std::vector<float>(cfg.obs_dim, 0.0f).data());
  reset();
}

const std::vector<float>& PolicyRunner::step(const float* observation) {
  const auto begin = Clock::now();
  torch::NoGradGuard no_grad;

  buffer_.push_observation(observation);

  const auto T = static_cast<long>(cfg_.context_len);
  const auto obs = torch::from_blob(const_cast<float*>(buffer_.observations().data()),
                                    {1, T, static_cast<long>(cfg_.obs_dim)})
                       .to(device_);
  const auto actions = torch::from_blob(const_cast<float*>(buffer_.actions().data()),
                                        {1, T, static_cast<long>(cfg_.act_dim)})
                           .to(device_);
  const auto timesteps = torch::from_blob(const_cast<long*>(buffer_.timesteps().data()), {1, T},
                                          torch::kLong)
                             .to(device_);

  const auto inference_begin = Clock::now();
  const auto output = module_.forward({timesteps, obs, actions}).toTensor().to(torch::kCPU);
  timing_.inference_us = micros_since(inference_begin);

  const auto* raw = output.contiguous().data_ptr<float>();
  for (std::size_t i = 0; i < cfg_.act_dim; ++i) {
    action_[i] = std::clamp(raw[i], -cfg_.action_clip, cfg_.action_clip);
  }

  // The unclipped action goes into the context, matching training, where the
  // env clipped on the way in but the buffer stored what the policy emitted.
  buffer_.push_action(raw);

  for (auto& a : action_) a *= cfg_.action_scale;
  timing_.total_us = micros_since(begin);
  return action_;
}

}  // namespace tert
