// Copyright (C) 2024-2025 Xiaozhe Wang <chaoslawful@gmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

#include "build_tools/template_instantiator.h"
#include "config.h"
#include "laplace/laplace_bem.impl.hcu"

HBEM_NS_OPEN

TEMPLATE_CLASS_EXPLICITLY_INSTANTIATE_4_WITH_RANGE_KERNEL_NUMBER_TYPES_(
  (LaplaceBEM),
  TI_INST_BOUNDARY_DIMS,
  TI_INST_SPACE_DIMS,
  TI_GENERATE_ALL_LAPLACE_TYPES_(TI_INST_NUM_TYPES));

HBEM_NS_CLOSE
