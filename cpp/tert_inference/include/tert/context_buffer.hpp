// Rolling observation-action history for the policy, mirroring the Python
// ContextWindow. Header-only and free of LibTorch so the indexing can be tested
// without a model or a GPU.
#pragma once

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace tert {

class ContextBuffer {
 public:
  ContextBuffer(std::size_t context_len, std::size_t obs_dim, std::size_t act_dim)
      : context_len_(context_len),
        obs_dim_(obs_dim),
        act_dim_(act_dim),
        obs_(context_len * obs_dim, 0.0f),
        actions_(context_len * act_dim, 0.0f),
        timesteps_(context_len, 0) {
    if (context_len == 0) throw std::invalid_argument("context_len must be positive");
  }

  // Zeroed history and timestep, as after a reset. The policy is trained on
  // this warm-start regime, so it is a valid input rather than a special case.
  void reset() {
    std::fill(obs_.begin(), obs_.end(), 0.0f);
    std::fill(actions_.begin(), actions_.end(), 0.0f);
    std::fill(timesteps_.begin(), timesteps_.end(), 0);
    step_ = 0;
  }

  void push_observation(const float* obs) {
    std::rotate(obs_.begin(), obs_.begin() + obs_dim_, obs_.end());
    std::copy(obs, obs + obs_dim_, obs_.end() - obs_dim_);
    refresh_timesteps();
  }

  void push_action(const float* action) {
    std::rotate(actions_.begin(), actions_.begin() + act_dim_, actions_.end());
    std::copy(action, action + act_dim_, actions_.end() - act_dim_);
    ++step_;
    refresh_timesteps();
  }

  const std::vector<float>& observations() const { return obs_; }
  const std::vector<float>& actions() const { return actions_; }
  const std::vector<long>& timesteps() const { return timesteps_; }

  std::size_t context_len() const { return context_len_; }
  std::size_t obs_dim() const { return obs_dim_; }
  std::size_t act_dim() const { return act_dim_; }
  long step() const { return step_; }

 private:
  // Slots before the episode began pin to 0; they carry zero observations
  // anyway, and a negative index would not embed.
  void refresh_timesteps() {
    for (std::size_t i = 0; i < context_len_; ++i) {
      const long offset = static_cast<long>(context_len_ - 1 - i);
      timesteps_[i] = std::max<long>(step_ - offset, 0);
    }
  }

  std::size_t context_len_, obs_dim_, act_dim_;
  std::vector<float> obs_, actions_;
  std::vector<long> timesteps_;
  long step_ = 0;
};

}  // namespace tert
