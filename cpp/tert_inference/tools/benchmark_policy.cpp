// Measures whether the exported policy holds the control deadline.
//
//   ./benchmark_policy policy.pt [steps]
//
// Mean latency is the uninteresting number. At 50 Hz the budget is 20 ms and a
// 3-block, 40-token model will not come close to it on average — what decides
// whether the robot walks is the tail, so this reports p99 and the worst step.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "tert/policy_runner.hpp"

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <policy.pt> [steps]\n", argv[0]);
    return 1;
  }
  const std::string model_path = argv[1];
  const int steps = argc > 2 ? std::atoi(argv[2]) : 1000;

  tert::RunnerConfig cfg;
  tert::PolicyRunner runner(model_path, cfg);

  std::mt19937 rng(0);
  std::normal_distribution<float> noise(0.0f, 1.0f);
  std::vector<float> observation(cfg.obs_dim);
  std::vector<double> latencies;
  latencies.reserve(steps);

  for (int i = 0; i < steps; ++i) {
    for (auto& value : observation) value = noise(rng);
    runner.step(observation.data());
    latencies.push_back(runner.timing().total_us);
  }

  std::sort(latencies.begin(), latencies.end());
  const double mean =
      std::accumulate(latencies.begin(), latencies.end(), 0.0) / static_cast<double>(steps);
  const auto at = [&](double q) { return latencies[static_cast<std::size_t>(q * (steps - 1))]; };

  std::printf("steps      %d\n", steps);
  std::printf("mean       %8.1f us\n", mean);
  std::printf("p50        %8.1f us\n", at(0.50));
  std::printf("p99        %8.1f us\n", at(0.99));
  std::printf("max        %8.1f us\n", latencies.back());
  std::printf("budget     %8.1f us  (50 Hz)\n", 20000.0);
  std::printf("%s\n", latencies.back() < 20000.0 ? "within budget" : "DEADLINE MISSED");
  return latencies.back() < 20000.0 ? 0 : 1;
}
