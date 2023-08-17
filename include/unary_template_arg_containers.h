/**
 * @file unary_template_arg_containers.h
 * @brief Definitions of STL container types with unary template argument.
 *
 * @date 2022-03-10
 * @author Jihuan Tian
 */
#ifndef INCLUDE_UNARY_TEMPLATE_ARG_CONTAINERS_H_
#define INCLUDE_UNARY_TEMPLATE_ARG_CONTAINERS_H_

#include <forward_list>
#include <list>
#include <vector>

namespace HierBEM
{
  template <typename T>
  using vector_uta = std::vector<T, std::allocator<T>>;

  template <typename T>
  using list_uta = std::list<T, std::allocator<T>>;

  template <typename T>
  using forward_list_uta = std::forward_list<T, std::allocator<T>>;
} // namespace HierBEM

#endif /* INCLUDE_UNARY_TEMPLATE_ARG_CONTAINERS_H_ */
