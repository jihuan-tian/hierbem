#include <cstdint>
#include <utility>
#include <vector>

/**
 * Partition the sequence with index range [0,n-1] into p partitions
 * such that the maximum cost over all partitions is minimized.
 *
 * The algorithm have the memory complexity of \f$O(np)\f$ and the time
 * complexity of \f$O(p(n-p))\f$.
 *
 * The cost function type is declared as template parameter to allow inlining it
 * by the compiler to achieve the maximum performance.
 *
 * NOTE: For large sequence the code compiled by Clang 12 with -O3/-Ofast optimization
 * can be ~50% faster than GCC 9.4 with the same flags.
 *
 * @ref Olstad and Manne (1995), "Efficient Partitioning of Sequences" (Fig. 1)
 */
template <typename IntervalCostFunc>
class SequencePartitioner
{
public:
  /**
   * Constructor for the sequence partitioner.
   *
   * @param n The length of the sequence.
   * @param p The number of partitions.
   * @param f The cost function for the interval [i,j]. This function should be of the form (int64_t i, int64_t j) -> double.
   */
  SequencePartitioner(int64_t n, int64_t p, const IntervalCostFunc &f)
    : _f(f)
    , _g(n * p)
    , _n(n)
    , _p(p)
    , _minmax_cost()
  {}
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
        for (; cur < _n && _f(prev, cur) <= _minmax_cost; cur++)
          ;
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
        // The first partition of optimal 1-partition for [i, n-1] is [i, n-1],
        // so g(i, 1) = f(i, n-1)
        _g[I(i, 1)] = _f(i, _n - 1);
      }

    for (int64_t k = 2; k <= _p; k++)
      {
        // The first partition for optimal k-partitions for [n-k, n-1] (the
        // interval length is k) is [n-k, n-k], as each partition has 1 element
        // at most
        _g[I(_n - k, k)] =
          std::max(_f(_n - k, _n - k), _g[I(_n - k + 1, k - 1)]);
        auto j = _n - k;

        for (int64_t i = _n - k - 1; i >= _p - k; i--)
          {
            // The maximum cost of the optimal k-partitions for [i, n-1] will be
            // found in [j+1, n-1], so g(i, k) = g(j+1, k-1) according to Lemma
            // 2 in the paper
            if (_f(i, j) <= _g[I(j + 1, k - 1)])
              {
                _g[I(i, k)] = _g[I(j + 1, k - 1)];
              }
            else
              {
                auto i_cost = _f(i, i);
                if (i_cost >= _g[I(j + 1, k - 1)])
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

                    // Current balance point is j, apply Theorem 5 in the paper
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
   * As the algorithm traverse memorizing matrix primarily in partition-wise
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
