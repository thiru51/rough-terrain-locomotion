// Plain asserts rather than a test framework: this has to build on a lab
// machine with no network, and the buffer has few enough invariants to check
// by hand.
#include "tert/context_buffer.hpp"

#include <cassert>
#include <cstdio>
#include <numeric>
#include <vector>

using tert::ContextBuffer;

namespace {

constexpr std::size_t kContext = 4, kObs = 3, kAct = 2;

void expect(bool condition, const char* what) {
  if (!condition) {
    std::fprintf(stderr, "FAILED: %s\n", what);
    std::abort();
  }
}

void push(ContextBuffer& buffer, float obs_value, float act_value) {
  const std::vector<float> obs(kObs, obs_value);
  const std::vector<float> act(kAct, act_value);
  buffer.push_observation(obs.data());
  buffer.push_action(act.data());
}

void starts_zeroed() {
  ContextBuffer buffer(kContext, kObs, kAct);
  expect(buffer.observations().size() == kContext * kObs, "observation buffer size");
  expect(buffer.actions().size() == kContext * kAct, "action buffer size");
  for (float v : buffer.observations()) expect(v == 0.0f, "observations start zeroed");
  expect(buffer.step() == 0, "step starts at zero");
}

void newest_entry_lands_last() {
  ContextBuffer buffer(kContext, kObs, kAct);
  push(buffer, 1.0f, 7.0f);

  const auto& obs = buffer.observations();
  expect(obs[obs.size() - 1] == 1.0f, "newest observation is in the last slot");
  expect(obs[0] == 0.0f, "older slots stay zero-padded at the front");
  expect(buffer.actions().back() == 7.0f, "newest action is in the last slot");
}

void oldest_entry_falls_off() {
  ContextBuffer buffer(kContext, kObs, kAct);
  for (std::size_t i = 1; i <= kContext + 1; ++i) push(buffer, static_cast<float>(i), 0.0f);

  const auto& obs = buffer.observations();
  expect(obs[0] == 2.0f, "the first push has been evicted");
  expect(obs[obs.size() - 1] == static_cast<float>(kContext + 1), "newest survives");
}

void timesteps_clamp_during_warm_start() {
  ContextBuffer buffer(kContext, kObs, kAct);
  push(buffer, 1.0f, 1.0f);

  const auto& t = buffer.timesteps();
  // One step taken: slots before the episode began pin to zero.
  expect(t[0] == 0 && t[1] == 0 && t[2] == 0, "pre-episode slots clamp to zero");
  expect(t[kContext - 1] == 1, "newest slot carries the current step");
}

void timesteps_advance_monotonically() {
  ContextBuffer buffer(kContext, kObs, kAct);
  for (int i = 0; i < 10; ++i) push(buffer, 0.0f, 0.0f);

  const auto& t = buffer.timesteps();
  expect(t[kContext - 1] == 10, "current step is the newest timestep");
  for (std::size_t i = 1; i < kContext; ++i) {
    expect(t[i] == t[i - 1] + 1, "timesteps are consecutive once past warm start");
  }
}

void reset_clears_history_and_time() {
  ContextBuffer buffer(kContext, kObs, kAct);
  for (int i = 0; i < 6; ++i) push(buffer, 5.0f, 5.0f);
  buffer.reset();

  expect(buffer.step() == 0, "reset clears the step counter");
  const auto& obs = buffer.observations();
  expect(std::accumulate(obs.begin(), obs.end(), 0.0f) == 0.0f, "reset zeroes observations");
  const auto& act = buffer.actions();
  expect(std::accumulate(act.begin(), act.end(), 0.0f) == 0.0f, "reset zeroes actions");
}

void observation_leads_action_within_a_step() {
  // The policy sees o_t before choosing a_t, so between the two pushes the
  // observation window is one entry ahead of the action window.
  ContextBuffer buffer(kContext, kObs, kAct);
  const std::vector<float> obs(kObs, 9.0f);
  buffer.push_observation(obs.data());

  expect(buffer.observations().back() == 9.0f, "observation is visible before the action");
  expect(buffer.actions().back() == 0.0f, "action slot is still unset");
  expect(buffer.step() == 0, "time only advances when the action is committed");
}

}  // namespace

int main() {
  starts_zeroed();
  newest_entry_lands_last();
  oldest_entry_falls_off();
  timesteps_clamp_during_warm_start();
  timesteps_advance_monotonically();
  reset_clears_history_and_time();
  observation_leads_action_within_a_step();
  std::puts("context buffer: all checks passed");
  return 0;
}
