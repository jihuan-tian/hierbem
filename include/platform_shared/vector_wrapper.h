// Copyright (C) 2023-2026 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

/**
 * @file vector_wrapper.h
 * @brief Definition of the class @p VectorWrapper.
 *
 * @date 2023-02-09
 * @author Jihuan Tian
 */
#ifndef HIERBEM_INCLUDE_LINEAR_ALGEBRA_VECTOR_WRAPPER_H_
#define HIERBEM_INCLUDE_LINEAR_ALGEBRA_VECTOR_WRAPPER_H_

#include <cassert>

#include "config.h"

HBEM_NS_OPEN

namespace PlatformShared
{
  /**
   * Simple vector class wrapping a pointer to an array of values, which can
   * be used both on the host and the device.
   */
  template <typename T = double>
  class VectorWrapper
  {
  public:
    using pointer         = T *;
    using const_pointer   = const T *;
    using reference       = T &;
    using const_reference = const T &;
    using size_type       = std::size_t;

    HBEM_ATTR_HOST HBEM_ATTR_DEV
    VectorWrapper();

    /**
     * Construct from a pointer allocated with memory.
     *
     * @param p Pointer to the array of values
     * @param _n Array size
     */
    HBEM_ATTR_HOST HBEM_ATTR_DEV
    VectorWrapper(pointer p, const size_type _n);

    /**
     * Reinitialize the current vector wrapper with a new pointer and size.
     */
    HBEM_ATTR_HOST HBEM_ATTR_DEV void
    reinit(pointer p, const size_type _n);

    // Copy constructor is deleted, because this wrapper class does not allocate
    // memory by itself.
    HBEM_ATTR_HOST HBEM_ATTR_DEV
    VectorWrapper(const VectorWrapper<T> &vec) = delete;

    HBEM_ATTR_HOST HBEM_ATTR_DEV
    VectorWrapper(VectorWrapper<T> &&vec);

    HBEM_ATTR_HOST HBEM_ATTR_DEV VectorWrapper<T>                              &
    operator=(const VectorWrapper<T> &vec);

    HBEM_ATTR_HOST HBEM_ATTR_DEV VectorWrapper<T>                              &
    operator=(VectorWrapper<T> &&vec);

    /**
     * Fill the vector from a source data pointer.
     */
    HBEM_ATTR_HOST HBEM_ATTR_DEV void
    fill(const size_type n_elements_for_copy,
         const_pointer   data,
         const size_type source_offset = 0,
         const size_type target_offset = 0);

    /**
     * Get the vector size.
     */
    HBEM_ATTR_HOST HBEM_ATTR_DEV size_type
    size() const;

    /**
     * Get the internal data pointer.
     */
    HBEM_ATTR_HOST HBEM_ATTR_DEV pointer
    data();

    /**
     * Get the internal data pointer (const version).
     */
    HBEM_ATTR_HOST HBEM_ATTR_DEV const_pointer
    data() const;

    /**
     * Get the reference to the i'th element in the vector.
     */
    HBEM_ATTR_HOST HBEM_ATTR_DEV reference
    operator()(const size_type i);

    /**
     * Get the reference to the i'th element in the vector.
     */
    HBEM_ATTR_HOST HBEM_ATTR_DEV reference
    operator[](const size_type i);

    /**
     * Get the const reference to the i'th element in the vector.
     */
    HBEM_ATTR_HOST HBEM_ATTR_DEV const_reference
    operator()(const size_type i) const;

    /**
     * Get the const reference to the i'th element in the vector.
     */
    HBEM_ATTR_HOST HBEM_ATTR_DEV const_reference
    operator[](const size_type i) const;

    /**
     * Print all values in the vector.
     */
    HBEM_ATTR_HOST HBEM_ATTR_DEV void
    print(const bool scientific = true) const;

    /**
     * Set all vector entries to zero.
     */
    HBEM_ATTR_HOST HBEM_ATTR_DEV void
    reinit();

  private:
    pointer   values;
    size_type n;
  };


  template <typename T>
  HBEM_ATTR_HOST HBEM_ATTR_DEV
  VectorWrapper<T>::VectorWrapper()
    : values(nullptr)
    , n(0)
  {}


  template <typename T>
  HBEM_ATTR_HOST HBEM_ATTR_DEV
  VectorWrapper<T>::VectorWrapper(pointer p, const size_type _n)
    : values(p)
    , n(_n)
  {}


  template <typename T>
  HBEM_ATTR_HOST HBEM_ATTR_DEV void
  VectorWrapper<T>::reinit(pointer p, const size_type _n)
  {
    values = p;
    n      = _n;
  }


  template <typename T>
  HBEM_ATTR_HOST HBEM_ATTR_DEV
  VectorWrapper<T>::VectorWrapper(VectorWrapper<T> &&vec)
    : values(vec.values)
    , n(vec.n)
  {}


  template <typename T>
  HBEM_ATTR_HOST HBEM_ATTR_DEV VectorWrapper<T> &
  VectorWrapper<T>::operator=(const VectorWrapper<T> &vec)
  {
    assert(n == vec.n);

    for (size_t i = 0; i < n; i++)
      values[i] = vec.values[i];

    return *this;
  }


  template <typename T>
  HBEM_ATTR_HOST HBEM_ATTR_DEV VectorWrapper<T> &
  VectorWrapper<T>::operator=(VectorWrapper<T> &&vec)
  {
    values = vec.values;
    n      = vec.n;

    return *this;
  }


  template <typename T>
  HBEM_ATTR_HOST HBEM_ATTR_DEV void
  VectorWrapper<T>::fill(const size_type n_elements_for_copy,
                         const_pointer   data,
                         const size_type source_offset,
                         const size_type target_offset)
  {
    [[maybe_unused]] const size_type allowed_n_elements_for_copy =
      n - target_offset;

    assert(n_elements_for_copy <= allowed_n_elements_for_copy);

    for (size_type i = 0; i < n_elements_for_copy; i++)
      values[target_offset + i] = data[source_offset + i];
  }


  template <typename T>
  HBEM_ATTR_HOST HBEM_ATTR_DEV inline typename VectorWrapper<T>::size_type
                 VectorWrapper<T>::size() const
  {
    return n;
  }


  template <typename T>
  HBEM_ATTR_HOST HBEM_ATTR_DEV inline typename VectorWrapper<T>::pointer
                 VectorWrapper<T>::data()
  {
    return values;
  }


  template <typename T>
  HBEM_ATTR_HOST HBEM_ATTR_DEV inline typename VectorWrapper<T>::const_pointer
                 VectorWrapper<T>::data() const
  {
    return values;
  }


  template <typename T>
  HBEM_ATTR_HOST HBEM_ATTR_DEV inline typename VectorWrapper<T>::reference
  VectorWrapper<T>::operator()(const size_type i)
  {
    return values[i];
  }


  template <typename T>
  HBEM_ATTR_HOST HBEM_ATTR_DEV inline typename VectorWrapper<T>::reference
  VectorWrapper<T>::operator[](const size_type i)
  {
    return values[i];
  }


  template <typename T>
  HBEM_ATTR_HOST HBEM_ATTR_DEV inline typename VectorWrapper<T>::const_reference
  VectorWrapper<T>::operator()(const size_type i) const
  {
    return values[i];
  }


  template <typename T>
  HBEM_ATTR_HOST HBEM_ATTR_DEV inline typename VectorWrapper<T>::const_reference
  VectorWrapper<T>::operator[](const size_type i) const
  {
    return values[i];
  }


  template <typename T>
  HBEM_ATTR_HOST HBEM_ATTR_DEV void
  VectorWrapper<T>::print(const bool scientific) const
  {
    for (size_type i = 0; i < n; i++)
      {
        if (scientific)
          printf("%20.8e\n", values[i]);
        else
          printf("%20.8f\n", values[i]);
      }
  }


  template <typename T>
  HBEM_ATTR_HOST HBEM_ATTR_DEV void
  VectorWrapper<T>::reinit()
  {
    for (size_type i = 0; i < n; i++)
      values[i] = 0;
  }
} // namespace PlatformShared

HBEM_NS_CLOSE

#endif // HIERBEM_INCLUDE_LINEAR_ALGEBRA_VECTOR_WRAPPER_H_
