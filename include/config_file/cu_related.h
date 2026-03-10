// Copyright (C) 2025 Xiaozhe Wang <chaoslawful@gmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

#ifndef HIERBEM_SRC_HBEM_LAPLACE_CU_RELATED_H
#define HIERBEM_SRC_HBEM_LAPLACE_CU_RELATED_H

#include "config.h"
#include "config_file/config_structs.h"

HBEM_NS_OPEN

void
initCudaRuntime(const ConfParallelization &parallel_params);

HBEM_NS_CLOSE

#endif // HIERBEM_SRC_HBEM_LAPLACE_CU_RELATED_H
