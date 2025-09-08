// Copyright (C) 2024 Xiaozhe Wang <chaoslawful@gmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

#include "build_tools/template_instantiator.h"
#include "sbb.impl.h"

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
