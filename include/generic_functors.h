/**
 * \file generic_functors.h
 * \brief This header file contains a set of self-defined generic functors.
 * \date 2021-07-20
 * \author Jihuan Tian
 */
#ifndef INCLUDE_GENERIC_FUNCTORS_H_
#define INCLUDE_GENERIC_FUNCTORS_H_

#include <deal.II/base/exceptions.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/types.h>

#include <deal.II/lac/lapack_support.h>
#include <deal.II/lac/vector.h>

#include <array>
#include <cmath>
#include <forward_list>
#include <iterator>
#include <map>
#include <vector>

using namespace dealii;

using size_type = std::make_unsigned<types::blas_int>::type;

/**
 * Find the pointer in a list of pointers of the same type. The comparison
 * is based on the data being pointed instead of the pointer addresses.
 * @param first
 * @param last
 * @param value_pointer
 * @return
 */
template <typename InputIterator, typename T>
InputIterator
find_pointer_data(InputIterator first,
                  InputIterator last,
                  const T *     value_pointer)
{
  for (; first != last; ++first)
    {
      /**
       * N.B. Here we use \p (*first) to get the pointer stored in the
       * container. Then use (*(*first)) to get the data associated with this
       * pointer.
       */
      if (*(*first) == (*value_pointer))
        {
          return first;
        }
    }

  return last;
}


/**
 * Permute a vector according to the permutation vector \p ipiv obtained from an
 * LU factorization.
 *
 * @param v
 * @param ipiv The vector storing the row permutation obtained from an LU
 * factorization. \alert{The row indices stored in \p ipiv starts from 1 instead
 * of 0.}
 */
template <typename VectorType>
void
permute_vector_by_ipiv(VectorType &                                v,
                       const std::vector<dealii::types::blas_int> &ipiv)
{
  AssertDimension(v.size(), ipiv.size());

  typename VectorType::value_type temp;
  for (std::size_t i = 0; i < ipiv.size(); i++)
    {
      temp           = v[i];
      v[i]           = v[ipiv[i] - 1];
      v[ipiv[i] - 1] = temp;
    }
}


/**
 * Copy a number of data from source vector to target vector.
 *
 * @param dst_vec
 * @param dst_start_index
 * @param src_vec
 * @param src_start_index
 * @param number_of_data
 */
template <typename number>
void
copy_vector(Vector<number> &                         dst_vec,
            const typename Vector<number>::size_type dst_start_index,
            const Vector<number> &                   src_vec,
            const typename Vector<number>::size_type src_start_index,
            const typename Vector<number>::size_type number_of_data)
{
  std::memcpy(dst_vec.data() + dst_start_index,
              src_vec.data() + src_start_index,
              number_of_data * sizeof(number));
}


/**
 * Get the size of a @p std::forward_list.
 *
 * @param fl
 * @return
 */
template <typename T>
size_type
size(const std::forward_list<T> &fl)
{
  return std::distance(fl.begin(), fl.end());
}


/**
 * Get the reference to the i'th element of the @p std::forward_list.
 *
 * @param fl
 * @param index
 * @return
 */
template <typename T>
T &
value_at(std::forward_list<T> &fl, const size_type i)
{
  typename std::forward_list<T>::iterator iter = fl.begin();
  std::advance(iter, i);

  return *iter;
}


/**
 * Get the const reference to the i'th element of the @p std::forward_list.
 *
 * @param fl
 * @param index
 * @return
 */
template <typename T>
const T &
value_at(const std::forward_list<T> &fl, const size_type i)
{
  typename std::forward_list<T>::const_iterator iter = fl.begin();
  std::advance(iter, i);

  return *iter;
}


/**
 * Erase the i'th element of the @p std::forward_list.
 *
 * @param fl
 * @param i
 */
template <typename T>
void
erase_at(std::forward_list<T> &fl, const size_type i)
{
  typename std::forward_list<T>::iterator iter = fl.before_begin();
  std::advance(iter, i);
  fl.erase_after(iter);
}


/**
 * Generate a list of linearized indices into a container by starting from the
 * given value with the specified increment step.
 *
 * \mynote{The definition of this function adopts the <code>template
 * template</code> and <code>alias template</code> techniques.}
 *
 * @param a The container holding the list of linearized indices
 * @param starting_value The starting value to be placed into the first element of the list
 * @param step The step value
 */
template <template <typename> typename Container, typename number>
void
gen_linear_indices(Container<number> &a,
                   number             starting_value = 0,
                   number             step           = 1)
{
  for (typename Container<number>::iterator iter = a.begin(); iter != a.end();
       std::advance(iter, 1), starting_value += step)
    {
      (*iter) = starting_value;
    }
}


/**
 * Compute the memory consumption for a @p std::map.
 *
 * @param m
 * @return
 */
template <typename KeyType, typename ValueType>
std::size_t
memory_consumption_of_map(const std::map<KeyType, ValueType> &m)
{
  return m.size() * (sizeof(KeyType) + sizeof(ValueType));
}


template <typename T>
std::size_t
memory_consumption_of_vector(const std::vector<T> &v)
{
  return v.capacity() * sizeof(T);
}


/**
 * Check the equality of the two points using raw comparison instead of
 * numerical comparison.
 *
 * \mynote{Here the raw comparison means each component in the @p dim
 * coordinates of @p p1 is directly compared with that of @p p2, which is
 * not checking if the absolute value of their difference is less than a
 * threshold. The latter is a common technique in numerical computation.}
 *
 * @param p1
 * @param p2
 * @return
 */
template <int spacedim, typename Number = double>
bool
is_equal(const Point<spacedim, Number> &p1, const Point<spacedim, Number> &p2)
{
  bool equality = true;

  for (unsigned int i = 0; i < spacedim; i++)
    {
      if (p1(i) != p2(i))
        {
          equality = false;
          break;
        }
    }

  return equality;
}


/**
 * Check the equality of the two points using numerical comparison instead of
 * raw comparison.
 *
 * @param p1
 * @param p2
 * @param threshold
 * @return
 */
template <int spacedim, typename Number = double>
bool
is_equal(const Point<spacedim, Number> &p1,
         const Point<spacedim, Number> &p2,
         const Number                   threshold)
{
  Assert(threshold > 0, ExcMessage("The given threshold value should be >0!"));

  bool equality = true;

  for (unsigned int i = 0; i < spacedim; i++)
    {
      if (std::abs(p1(i) - p2(i)) > threshold)
        {
          equality = false;
          break;
        }
    }

  return equality;
}


/**
 * Convert a vector to a rank-1 tensor.
 *
 * @param v
 * @param t
 */
template <int dim, typename Number, typename VectorType>
void
VectorToTensor(const VectorType &v, Tensor<1, dim, Number> &t)
{
  AssertDimension(v.size(), dim);

  for (unsigned int i = 0; i < dim; i++)
    {
      t[i] = v[i];
    }
}


template <int dim, typename Number, typename VectorType>
Tensor<1, dim, Number>
VectorToTensor(const VectorType &v)
{
  AssertDimension(v.size(), dim);

  Tensor<1, dim, Number> t;

  for (unsigned int i = 0; i < dim; i++)
    {
      t[i] = v[i];
    }

  return t;
}


/**
 * Check if @p range1 is a subset of @p range2.
 *
 * @param range1
 * @param range2
 * @return
 */
template <typename Value>
bool
is_subset(const std::array<Value, 2> &range1,
          const std::array<Value, 2> &range2)
{
  return ((range1[0] >= range2[0]) && (range1[1] <= range2[1]));
}


/**
 * Check if @p range1 is a proper subset of @p range2.
 *
 * @param range1
 * @param range2
 * @return
 */
template <typename Value>
bool
is_proper_subset(const std::array<Value, 2> &range1,
                 const std::array<Value, 2> &range2)
{
  return (((range1[0] >= range2[0]) && (range1[1] < range2[1])) ||
          ((range1[0] > range2[0]) && (range1[1] <= range2[1])));
}


/**
 * Check if @p range1 is a super set of @p range2.
 *
 * @param range1
 * @param range2
 * @return
 */
template <typename Value>
bool
is_superset(const std::array<Value, 2> &range1,
            const std::array<Value, 2> &range2)
{
  return ((range1[0] <= range2[0]) && (range1[1] >= range2[1]));
}


/**
 * Check if @p range1 is a proper super set of @p range2.
 *
 * @param range1
 * @param range2
 * @return
 */
template <typename Value>
bool
is_proper_superset(const std::array<Value, 2> &range1,
                   const std::array<Value, 2> &range2)
{
  return (((range1[0] <= range2[0]) && (range1[1] > range2[1])) ||
          ((range1[0] < range2[0]) && (range1[1] >= range2[1])));
}


template <typename Value>
void
intersect(const std::array<Value, 2> &range1,
          const std::array<Value, 2> &range2,
          std::array<Value, 2> &      range_intersection)
{
  range_intersection[0] = 0;
  range_intersection[1] = 0;

  types::global_dof_index larger_lower_bound  = std::max(range1[0], range2[0]);
  types::global_dof_index smaller_upper_bound = std::min(range1[1], range2[1]);

  if (smaller_upper_bound > larger_lower_bound)
    {
      // When a non-empty range is obtained.
      range_intersection[0] = larger_lower_bound;
      range_intersection[1] = smaller_upper_bound;
    }
}

#endif /* INCLUDE_GENERIC_FUNCTORS_H_ */
