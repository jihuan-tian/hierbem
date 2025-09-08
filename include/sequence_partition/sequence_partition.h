// Copyright (C) 2024 Xiaozhe Wang <chaoslawful@gmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

#ifndef HIERBEM_INCLUDE_SEQUENCE_PARTITION_SEQUENCE_PARTITION_H_
#define HIERBEM_INCLUDE_SEQUENCE_PARTITION_SEQUENCE_PARTITION_H_

#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include "config.h"

HBEM_NS_OPEN

/**
 * Partition the sequence with index range [0,n-1] into p partitions
 * such that the maximum cost over all partitions is minimized.
 *
 * The algorithm have the memory complexity of \f$O(np)\f$ and the time
 * complexity of \f$O(p(n-p))\f$.
 *
 * The cost function type is declared as template parameter to allow inlining
 * it by the compiler to achieve the maximum performance.
 *
 * NOTE: For large sequence the code compiled by Clang 12 with -O3/-Ofast
 * optimization can be ~50% faster than GCC 9.4 with the same flags.
 *
 * \myref{Olstad and Manne (1995), "Efficient Partitioning of Sequences" (Fig.
 * 1)}
 */
template <typename IntervalCostFunc>
class SequencePartitioner
{
public:
  /**
   * Constructor for the sequence partitioner.
   *
   * @param n The length of the sequence. (n >= 1)
   * @param p The number of partitions. (1 <= p <= n)
   * @param f The cost function for the interval [i,j]. This function should be
   * of the form (int64_t i, int64_t j) -> double.
   */
  SequencePartitioner(int64_t n, int64_t p, const IntervalCostFunc &f)
    : _f(f)
    , _g(n * p)
    , _n(n)
    , _p(p)
    , _minmax_cost()
  {
    if (n < 1)
      {
        throw std::invalid_argument("The length of the sequence must be >=1: " +
                                    std::to_string(n));
      }
    if (p < 1 || p > n)
      {
        throw std::invalid_argument(
          "The number of expected partitions must be >=1 and <=n: " +
          std::to_string(p));
      }
  }
  ~SequencePartitioner() = default;

  /**
   * Get the minimum maximum cost of the partitioning.
   */
  double
  get_minmax_cost() const
  {
    return _minmax_cost;
  }

  /**
   * Get all partition intervals by linear search.
   *
   * @param parts The vector to store the partition intervals.
   */
  void
  get_partitions(std::vector<std::pair<int64_t, int64_t>> &parts) const
  {
    parts.resize(_p);

    int64_t cur  = 0;
    int64_t prev = 0;
    for (int64_t k = 0; k < _p - 1; k++)
      {
        while (cur < _n - (_p - (k + 1)) // ensure there are enough indexes
                                         // left for the rest of partitions
               && _f(prev, cur) <= _minmax_cost // cost of current interval is
                                                // no more than the maximum cost
        )
          {
            ++cur; // extend right boundary of current interval
          }
        parts[k] = std::make_pair(prev, cur - 1);
        prev     = cur;
      }
    parts[_p - 1] = std::make_pair(prev, _n - 1);
  }

  /**
   * Partition the interval [0, n-1] into <i>p</i> partitions such that the
   * maximum cost over all partitions is minimized.
   */
  void
  partition()
  {
    for (int64_t i = _n - 1; i >= _p - 1; i--)
      {
        // The first partition of optimal 1-partition for [i, n-1] is [i,
        // n-1], so g(i, 1) = f(i, n-1)
        _g[I(i, 1)] = _f(i, _n - 1);
      }

    for (int64_t k = 2; k <= _p; k++)
      {
        // The first partition for optimal k-partitions for [n-k, n-1] (the
        // interval length is k) is [n-k, n-k], as each partition has 1
        // element at most
        _g[I(_n - k, k)] =
          std::max(_f(_n - k, _n - k), _g[I(_n - k + 1, k - 1)]);
        auto j = _n - k;

        for (int64_t i = _n - k - 1; i >= _p - k; i--)
          {
            // The maximum cost of the optimal k-partitions for [i, n-1] will
            // be found in [j+1, n-1], so g(i, k) = g(j+1, k-1) according to
            // Lemma 2 in the paper
            if (_f(i, j) <= _g[I(j + 1, k - 1)])
              {
                _g[I(i, k)] = _g[I(j + 1, k - 1)];
              }
            else
              {
                auto i_cost = _f(i, i);
                if (i_cost >= _g[I(i + 1, k - 1)])
                  {
                    _g[I(i, k)] = i_cost;
                    j           = i;
                  }
                else
                  {
                    // Search for balance point in the interval [i, j]
                    while (_f(i, j - 1) >= _g[I(j, k - 1)])
                      {
                        j--;
                      }

                    // Current balance point is j, apply Theorem 5 in the
                    // paper
                    _g[I(i, k)] = std::min(_f(i, j), _g[I(j, k - 1)]);
                    if (_g[I(i, k)] == _g[I(j, k - 1)])
                      {
                        j--;
                      }
                  }
              }
          }
      }

    _minmax_cost = _g[I(0, _p)];
  }

private:
  /**
   * 2D memorizing matrix to underlying storage index mapping.
   *
   * As the algorithm traverse memorizing matrix primarily in starting index
   * order, the column-wise storage is more cache-friendly than the row-wise
   * (~10x faster).
   *
   * @param i The starting index in sequence (0 to n-1)
   * @param k The targeting partition number (1 to p)
   * @returns The 1D index in the column-wise storage vector.
   */
  inline int64_t
  I(int64_t i, int64_t k) const
  {
    return (k - 1) * _n + i;
  }

  const IntervalCostFunc &_f;
  std::vector<double>     _g; /** g(i,p) optimal cost memorizing matrix */
  int64_t                 _n;
  int64_t                 _p;
  double                  _minmax_cost;
};

HBEM_NS_CLOSE

#endif // HIERBEM_INCLUDE_SEQUENCE_PARTITION_SEQUENCE_PARTITION_H_
