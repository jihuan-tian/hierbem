/**
 * @file cpu_table.h
 * @brief Introduction of cpu_table.h
 *
 * @date 2023-01-25
 * @author Jihuan Tian
 */
#ifndef HIERBEM_INCLUDE_LINEAR_ALGEBRA_CPU_TABLE_H_
#define HIERBEM_INCLUDE_LINEAR_ALGEBRA_CPU_TABLE_H_

#include <deal.II/base/table.h>
#include <deal.II/base/table_indices.h>

#include "config.h"

HBEM_NS_OPEN

using namespace dealii;

/**
 * The main purpose of this class is to expose the internal data pointer
 * within @p Table to the outside.
 *
 * @tparam N Table dimension
 * @tparam T Value type
 */
template <int N, typename T>
class CPUTable : public TableBase<N, T>
{
public:
  /**
   * Default constructor.
   */
  CPUTable() = default;

  /**
   * Constructor by specifying the table sizes.
   *
   * @param sizes
   */
  CPUTable(const TableIndices<N> &sizes);

  /**
   * Constructor by specifying the table sizes and initializing it with data
   * from a linear data.
   *
   * @param sizes
   * @param entries
   * @param C_style_indexing
   */
  template <typename InputIterator>
  CPUTable(const TableIndices<N> &sizes,
           InputIterator          entries,
           const bool             C_style_indexing = true);

  /**
   * Get the pointer to the internal data held within the table.
   *
   * @return
   */
  typename AlignedVector<T>::pointer
  data();

  /**
   * Get the const pointer to the internal data held within the table.
   *
   * @return
   */
  typename AlignedVector<T>::const_pointer
  data() const;
};


template <int N, typename T>
CPUTable<N, T>::CPUTable(const TableIndices<N> &sizes)
  : TableBase<N, T>(sizes)
{}


template <int N, typename T>
template <typename InputIterator>
CPUTable<N, T>::CPUTable(const TableIndices<N> &sizes,
                         InputIterator          entries,
                         const bool             C_style_indexing)
  : TableBase<N, T>(sizes, entries, C_style_indexing)
{}


template <int N, typename T>
typename AlignedVector<T>::pointer
CPUTable<N, T>::data()
{
  return TableBase<N, T>::values.data();
}


template <int N, typename T>
typename AlignedVector<T>::const_pointer
CPUTable<N, T>::data() const
{
  return TableBase<N, T>::values.data();
}

HBEM_NS_CLOSE

#endif // HIERBEM_INCLUDE_LINEAR_ALGEBRA_CPU_TABLE_H_
