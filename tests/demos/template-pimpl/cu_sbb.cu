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
#include "cu_sbb.impl.h"

#define INST_CLASSES (CuSbb)

TEMPLATE_CLASS_EXPLICITLY_INSTANTIATE_1(INST_CLASSES, TI_INST_NUM_TYPES)
