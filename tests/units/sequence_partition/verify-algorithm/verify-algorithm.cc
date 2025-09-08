// Copyright (C) 2024 Xiaozhe Wang <chaoslawful@gmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

/**
 * @file verify-algorithm.cc
 * @brief
 *
 * @ingroup test_cases
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

int
generate_randint(int seed, int a, int b)
{
  std::mt19937                    gen(seed);
  std::uniform_int_distribution<> dis(a, b);
  return dis(gen);
}

TEST_CASE("Sequence Partition Basic Test", "[seq_part]")
{
  int  n            = 40;
  int  n_partitions = 5;
  auto trial_no     = GENERATE(range(0, FUZZING_TIMES));
  SECTION(std::string("trial #") + std::to_string(trial_no))
  {
    auto task_costs = generate_random_task_costs(FIXED_SEED + trial_no, n);
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
  auto trial_no = GENERATE(range(0, FUZZING_TIMES));

  SECTION(std::string("trial #") + std::to_string(trial_no))
  {
    int n = (trial_no + 1) * 10;
    int p = generate_randint(FIXED_SEED + trial_no, 1, n);
    INFO("n=" << n << ", p=" << p);

    auto task_costs = generate_random_task_costs(FIXED_SEED + trial_no, n);
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

TEST_CASE("Non-empty partition and correct minmax cost test (p == n)",
          "[seq_part]")
{
  auto trial_no = GENERATE(range(0, FUZZING_TIMES));

  SECTION(std::string("trial #") + std::to_string(trial_no))
  {
    int n = (trial_no + 1) * 10;
    int p = n;
    INFO("n=" << n << ", p=" << p);

    auto task_costs = generate_random_task_costs(FIXED_SEED + trial_no, n);
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

TEST_CASE("Non-empty partition and correct minmax cost test (p == 1)",
          "[seq_part]")
{
  auto trial_no = GENERATE(range(0, FUZZING_TIMES));

  SECTION(std::string("trial #") + std::to_string(trial_no))
  {
    int n = (trial_no + 1) * 10;
    int p = 1;
    INFO("n=" << n << ", p=" << p);

    auto task_costs = generate_random_task_costs(FIXED_SEED + trial_no, n);
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

TEST_CASE("Invalid arguments", "[seq_part]")
{
  int  n        = 10;
  auto trial_no = GENERATE(range(0, FUZZING_TIMES));
  SECTION(std::string("trial #") + std::to_string(trial_no))
  {
    auto task_costs = generate_random_task_costs(FIXED_SEED + trial_no, n);
    auto cost_func  = [&task_costs](int i, int j) -> double {
      double sum = 0.0;
      for (int k = i; k <= j; k++)
        {
          sum += task_costs[k];
        }
      return sum;
    };

    // invalid sequence length (<=0)
    REQUIRE_THROWS([&]() {
      SequencePartitioner<decltype(cost_func)> sp(-trial_no, 1, cost_func);
    }());

    // invalid partition number (<=0)
    REQUIRE_THROWS([&]() {
      SequencePartitioner<decltype(cost_func)> sp(n, -trial_no, cost_func);
    }());

    // invalid partition number (>n)
    REQUIRE_THROWS([&]() {
      SequencePartitioner<decltype(cost_func)> sp(n,
                                                  n + trial_no + 1,
                                                  cost_func);
    }());
  }
}
