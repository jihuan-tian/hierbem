#include <chrono>
#include <execution>
#include <iostream>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

#include "sequence_partition/sequence_partition.h"

using namespace HierBEM;

int
main()
{
  std::cout << "Generating random costs" << std::endl;
  std::random_device               rd;
  std::mt19937                     gen(rd());
  std::uniform_real_distribution<> dis(0.0, 1.0);

  int64_t n = 1000000;
  int64_t p = 256;

  // std::vector<double> task_costs(n);
  // for (auto &num : task_costs)
  //   {
  //     num = dis(gen);
  //   }

  // auto cost_func = [&task_costs](int64_t i, int64_t j) -> double {
  //   return std::accumulate(task_costs.begin() + i, task_costs.begin() + j,
  //   0.0);
  // };

  auto cost_func = [](int64_t i, int64_t j) -> double { return j - i + 1.0; };

  std::cout << "Optimizing partitioning" << std::endl;
  SequencePartitioner<decltype(cost_func)> sp(n, p, cost_func);

  auto start = std::chrono::high_resolution_clock::now();
  sp.partition();
  auto end = std::chrono::high_resolution_clock::now();

  double minmax_cost = sp.get_minmax_cost();
  std::cout << "Minimum maximum interval cost: " << minmax_cost << std::endl;
  std::cout << "Time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                     start)
                 .count()
            << " ms" << std::endl;

  std::vector<std::pair<int64_t, int64_t>> parts;
  start = std::chrono::high_resolution_clock::now();
  sp.get_partitions(parts);
  end = std::chrono::high_resolution_clock::now();
  std::cout << "Traverse partitions in "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                     start)
                 .count()
            << " ms: " << std::endl;

  double max_cost = 0.0;
  for (int64_t i = 0; i < p; i++)
    {
      int64_t left          = parts[i].first;
      int64_t right         = parts[i].second;
      double  interval_cost = cost_func(left, right);
#if 0
      std::cout << "[" << left << "," << right << "]: " << interval_cost
                << std::endl;
#endif
      if (interval_cost > max_cost)
        {
          max_cost = interval_cost;
        }
    }
  std::cout << "maximum cost: " << max_cost << std::endl;

  return 0;
}
