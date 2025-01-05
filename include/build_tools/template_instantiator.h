#ifndef INCLUDE_TEMPLATE_INSTANTIATOR_H_
#define INCLUDE_TEMPLATE_INSTANTIATOR_H_

/**
 * Helper macros to ease instantiate template classes for different
 * combinations of template arguments.
 * */

#include <boost/preprocessor.hpp>

/* Internal macros */
#define TI_ARG0_(tup) BOOST_PP_SEQ_ELEM(0, tup)
#define TI_ARG1_(tup) BOOST_PP_SEQ_ELEM(1, tup)
#define TI_ARG2_(tup) BOOST_PP_SEQ_ELEM(2, tup)

#define TI_INSTANTIATE_CLS_1_(r, prod) \
  template class TI_ARG0_(prod)<TI_ARG1_(prod)>;
#define TI_INSTANTIATE_CLS_2_(r, prod) \
  template class TI_ARG0_(prod)<TI_ARG1_(prod), TI_ARG2_(prod)>;

#define TEMPLATE_CLASS_EXPLICITLY_INSTANTIATE_1(cls_seq, arg1_seq) \
  BOOST_PP_SEQ_FOR_EACH_PRODUCT(TI_INSTANTIATE_CLS_1_, (cls_seq)(arg1_seq))
#define TEMPLATE_CLASS_EXPLICITLY_INSTANTIATE_2(cls_seq, arg1_seq, arg2_seq) \
  BOOST_PP_SEQ_FOR_EACH_PRODUCT(TI_INSTANTIATE_CLS_2_,                       \
                                (cls_seq)(arg1_seq)(arg2_seq))

#define TI_INST_NUM_TYPES (float)(double)
#define TI_INST_SPACE_DIMS (3)
#define TI_INST_BOUNDARY_DIMS (2)

#endif // INCLUDE_TEMPLATE_INSTANTIATOR_H_
