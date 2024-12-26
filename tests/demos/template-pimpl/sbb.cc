#include "sbb.impl.h"
#include "build/template_instantiator.h"

#define INST_CLASSES (SBB)
#define INST_DIMS (2)(3)
#define INST_TYPES (float)(double)

TEMPLATE_CLASS_EXPLICITLY_INSTANTIATE_2(INST_CLASSES, INST_DIMS, INST_TYPES)

#define EXPAND_INST_(r, prod)                        \
  template SBB<TI_ARG1_(prod), TI_ARG2_(prod)>::SBB( \
    const Mapping<TI_ARG0_(prod), TI_ARG1_(prod)> &);

#define INST_MAP_DIMS (1)(2)(3)
BOOST_PP_SEQ_FOR_EACH_PRODUCT(EXPAND_INST_,
                              (INST_MAP_DIMS)(INST_DIMS)(INST_TYPES))
