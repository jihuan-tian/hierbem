// Copyright (C) 2024 Xiaozhe Wang <chaoslawful@gmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

#ifndef CU_SBB_H__
#define CU_SBB_H__

template <typename NumType>
class CuSbb
{
public:
  void
  vec_add(NumType *a, NumType *b, NumType *c, int n);
};


#endif // CU_SBB_H__
