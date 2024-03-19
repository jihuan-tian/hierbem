/**
 * @file verify-algorithm.cc
 * @brief
 *
 * @ingroup testers
 * @author
 * @date 2024-02-04
 */
#include <catch2/catch_all.hpp>

#include <random>
#include <vector>

#include "sequence_partition/sequence_partition.h"

using namespace Catch::Matchers;
using namespace HierBEM;

static constexpr int FUZZING_TIMES = 5;
static constexpr int FIXED_SEED    = 123456;

std::vector<double>
generate_random_task_costs(int seed, int n)
{
  std::mt19937                     gen(seed);
  std::uniform_real_distribution<> dis(0.0, 1.0);

  std::vector<double> task_costs(n);
  for (auto &num : task_costs)
    {
      num = dis(gen);
    }
  return task_costs;
}

TEST_CASE("Sequence Partition Basic Test", "[seq_part]")
{
  int  n            = 40;
  int  n_partitions = 5;
  auto trial_no     = GENERATE(range(0, FUZZING_TIMES));
  SECTION(std::string("trial #") + std::to_string(trial_no))
  {
    auto task_costs = generate_random_task_costs(FIXED_SEED, n);
    auto cost_func  = [&task_costs](int i, int j) -> double {
      double sum = 0.0;
      for (int k = i; k <= j; k++)
        {
          sum += task_costs[k];
        }
      return sum;
    };

    SequencePartitioner<decltype(cost_func)> sp(n, n_partitions, cost_func);
    sp.partition();

    double minmax_cost = sp.get_minmax_cost();
    CHECK(minmax_cost > 0.0);
  }
}

TEST_CASE("Partition minmax cost non-increasing to the number of partitions",
          "[seq_part]")
{
  int n              = 40;
  int max_partitions = 10;

  auto   task_costs = generate_random_task_costs(FIXED_SEED, n);
  double prev_cost  = 0.0;
  for (int p = 1; p <= max_partitions; p++)
    {
      auto cost_func = [&task_costs](int i, int j) -> double {
        double sum = 0.0;
        for (int k = i; k <= j; k++)
          {
            sum += task_costs[k];
          }
        return sum;
      };
      SequencePartitioner<decltype(cost_func)> sp(n, p, cost_func);
      sp.partition();

      double minmax_cost = sp.get_minmax_cost();
      CHECK(minmax_cost > 0.0);

      INFO("Minimum maximum interval cost for "
           << p << " partitions: " << minmax_cost);

      // Non-increasing property test
      if (prev_cost > 0.0)
        {
          CHECK(minmax_cost <= prev_cost);
        }
      else
        {
          prev_cost = minmax_cost;
        }
    }
}

TEST_CASE("Non-empty partition and correct minmax cost test", "[seq_part]")
{
  int  n        = 40;
  int  p        = 10;
  auto trial_no = GENERATE(range(0, FUZZING_TIMES));

  SECTION(std::string("trial #") + std::to_string(trial_no))
  {
    auto task_costs = generate_random_task_costs(FIXED_SEED, n);
    auto cost_func  = [&task_costs](int i, int j) -> double {
      double sum = 0.0;
      for (int k = i; k <= j; k++)
        {
          sum += task_costs[k];
        }
      return sum;
    };

    SequencePartitioner<decltype(cost_func)> sp(n, p, cost_func);
    sp.partition();

    double pred_cost   = sp.get_minmax_cost();
    double actual_cost = 0.0;

    std::vector<std::pair<int64_t, int64_t>> parts;
    sp.get_partitions(parts);

    for (auto &part : parts)
      {
        // Make sure each partition is not empty
        CHECK(part.first <= part.second);
        actual_cost = std::max(actual_cost, cost_func(part.first, part.second));
      }

    // Make sure the actual cost is the same as the predicted cost
    REQUIRE_THAT(actual_cost, WithinAbs(pred_cost, 1e-6));
  }
}

TEST_CASE("Non-empty partition and correct minmax cost test (n == p)",
          "[seq_part]")
{
  int  n        = 40;
  int  p        = 40;
  auto trial_no = GENERATE(range(0, FUZZING_TIMES));

  SECTION(std::string("trial #") + std::to_string(trial_no))
  {
    auto task_costs = generate_random_task_costs(FIXED_SEED, n);
    auto cost_func  = [&task_costs](int i, int j) -> double {
      double sum = 0.0;
      for (int k = i; k <= j; k++)
        {
          sum += task_costs[k];
        }
      return sum;
    };

    SequencePartitioner<decltype(cost_func)> sp(n, p, cost_func);
    sp.partition();

    double pred_cost   = sp.get_minmax_cost();
    double actual_cost = 0.0;

    std::vector<std::pair<int64_t, int64_t>> parts;
    sp.get_partitions(parts);

    for (auto &part : parts)
      {
        // Make sure each partition is not empty
        CHECK(part.first <= part.second);
        actual_cost = std::max(actual_cost, cost_func(part.first, part.second));
      }

    // Make sure the actual cost is the same as the predicted cost
    REQUIRE_THAT(actual_cost, WithinAbs(pred_cost, 1e-6));
  }
}
