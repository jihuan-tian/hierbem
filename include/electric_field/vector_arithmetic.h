/**
 * @file vector_arithmetic.h
 * @brief Introduction of vector_arithmetic.h
 *
 * @date 2024-08-05
 * @author Jihuan Tian
 */
#ifndef INCLUDE_ELECTRIC_FIELD_VECTOR_ARITHMETIC_H_
#define INCLUDE_ELECTRIC_FIELD_VECTOR_ARITHMETIC_H_

#include <deal.II/base/exceptions.h>

#include <vector>

namespace HierBEM
{
  using namespace dealii;

  /**
   * Compute \f$x = x + \alpha y\f$, where both \f$x\f$ and \f$y\f$ are
   * @p std::vector.
   *
   * @pre
   * @post
   * @param x
   * @param y
   * @param alpha
   */
  template <typename Value>
  void
  add_vector(std::vector<Value>       &x,
             const std::vector<Value> &y,
             const Value               alpha)
  {
    AssertDimension(x.size(), y.size());

    for (size_t i = 0; i < x.size(); i++)
      {
        x[i] += alpha * y[i];
      }
  }

  /**
   * Compute \f$x = \alpha \cdot x\f$
   *
   * @pre
   * @post
   * @param x
   * @param alpha
   */
  template <typename Value>
  void
  scale_vector(std::vector<Value> &x, const Value alpha)
  {
    for (auto &v : x)
      {
        v *= alpha;
      }
  }

  template <typename Value>
  void
  normalize_vector(std::vector<Value> &x)
  {
    Value magnitude = Value();
    for (auto v : x)
      magnitude += v * v;
    magnitude = std::sqrt(magnitude);

    scale_vector(x, magnitude);
  }
} // namespace HierBEM

#endif /* INCLUDE_ELECTRIC_FIELD_VECTOR_ARITHMETIC_H_ */
