// Copyright (C) 2024-2025 Xiaozhe Wang <chaoslawful@gmail.com>
// Copyright (C) 2025 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

#ifndef INCLUDE_BUILD_TOOLS_TEMPLATE_INSTANTIATOR_H_
#define INCLUDE_BUILD_TOOLS_TEMPLATE_INSTANTIATOR_H_

/**
 * Helper macros to ease instantiate template classes for different
 * combinations of template arguments.
 * */

#include <boost/preprocessor.hpp>

#include <complex>

/* Internal macros */
#define TI_ARG0_(tup) BOOST_PP_SEQ_ELEM(0, tup)
#define TI_ARG1_(tup) BOOST_PP_SEQ_ELEM(1, tup)
#define TI_ARG2_(tup) BOOST_PP_SEQ_ELEM(2, tup)
#define TI_ARG3_(tup) BOOST_PP_SEQ_ELEM(3, tup)

#define TI_INSTANTIATE_CLS_1_(r, prod) \
  template class TI_ARG0_(prod)<TI_ARG1_(prod)>;
#define TI_INSTANTIATE_CLS_2_(r, prod) \
  template class TI_ARG0_(prod)<TI_ARG1_(prod), TI_ARG2_(prod)>;
#define TI_INSTANTIATE_CLS_4_WITH_RANGE_KERNEL_NUMBER_TYPES_(r, prod) \
  template class TI_ARG0_(prod)<TI_ARG1_(prod),                       \
                                TI_ARG2_(prod),                       \
                                TI_ARG0_(TI_ARG3_(prod)),             \
                                TI_ARG1_(TI_ARG3_(prod))>;

#define TEMPLATE_CLASS_EXPLICITLY_INSTANTIATE_1(cls_seq, arg1_seq) \
  BOOST_PP_SEQ_FOR_EACH_PRODUCT(TI_INSTANTIATE_CLS_1_, (cls_seq)(arg1_seq))
#define TEMPLATE_CLASS_EXPLICITLY_INSTANTIATE_2(cls_seq, arg1_seq, arg2_seq) \
  BOOST_PP_SEQ_FOR_EACH_PRODUCT(TI_INSTANTIATE_CLS_2_,                       \
                                (cls_seq)(arg1_seq)(arg2_seq))
#define TEMPLATE_CLASS_EXPLICITLY_INSTANTIATE_4_WITH_RANGE_KERNEL_NUMBER_TYPES_( \
  cls_seq, arg1_seq, arg2_seq, arg3_seq)                                         \
  BOOST_PP_SEQ_FOR_EACH_PRODUCT(                                                 \
    TI_INSTANTIATE_CLS_4_WITH_RANGE_KERNEL_NUMBER_TYPES_,                        \
    (cls_seq)(arg1_seq)(arg2_seq)(arg3_seq))

#define TI_INST_NUM_TYPES (float)(double)
#define TI_INST_SPACE_DIMS (3)
#define TI_INST_BOUNDARY_DIMS (2)

/* Generate compatible pairs of RangeNumberType and KernelNumberType for an
 * underlying scalar type for the Laplace solver */
#define TI_GENERATE_LAPLACE_TYPES_(r, data, scalar_type) \
  ((std::complex<scalar_type>)(scalar_type))((scalar_type)(scalar_type))

/* Generate compatible pairs of RangeNumberType and KernelNumberType for an
 * underlying scalar type for the Helmholtz solver */
#define TI_GENERATE_HELMHOLTZ_TYPES_(r, data, scalar_type) \
  ((std::complex<scalar_type>)(std::complex<scalar_type>))

/* Generate a sequence of compatible pairs of RangeNumberType and
 * KernelNumberType for all underlying scalar types for the Laplace solver. */
#define TI_GENERATE_ALL_LAPLACE_TYPES_(scalar_seq) \
  BOOST_PP_SEQ_FOR_EACH(TI_GENERATE_LAPLACE_TYPES_, _, scalar_seq)

/* Generate a sequence of compatible pairs of RangeNumberType and
 * KernelNumberType for all underlying scalar types for the Helmholtz solver. */
#define TI_GENERATE_ALL_HELMHOLTZ_TYPES_(scalar_seq) \
  BOOST_PP_SEQ_FOR_EACH(TI_GENERATE_HELMHOLTZ_TYPES_, _, scalar_seq)

#endif // INCLUDE_BUILD_TOOLS_TEMPLATE_INSTANTIATOR_H_
