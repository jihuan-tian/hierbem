/**
 * \file generic_functors.h
 * \brief This header file contains a set of self-defined generic functors.
 * \date 2021-07-20
 * \author Jihuan Tian
 */
#ifndef INCLUDE_GENERIC_FUNCTORS_H_
#define INCLUDE_GENERIC_FUNCTORS_H_

#include <deal.II/base/types.h>

#include <map>
#include <vector>

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
 * Build a map from global indices to local indices based on the given index
 * set, which is actually a map from local indices to global indices.
 * @param index_set_as_local_to_global_map
 * @param global_to_local_map
 */
void
build_index_set_global_to_local_map(
  const std::vector<dealii::types::global_dof_index>
    &index_set_as_local_to_global_map,
  std::map<dealii::types::global_dof_index, size_t> &global_to_local_map);

#endif /* INCLUDE_GENERIC_FUNCTORS_H_ */
