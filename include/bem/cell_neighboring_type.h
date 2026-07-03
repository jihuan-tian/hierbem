// Copyright (C) 2026 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

/**
 * @file cell_neighboring_type.h
 * @brief Definition of enums for cell neighboring types and their detection
 * methods.
 *
 * @date 2026-05-12
 * @author Jihuan Tian
 */
#ifndef HIERBEM_INCLUDE_BEM_CELL_NEIGHBORING_TYPE_H_
#define HIERBEM_INCLUDE_BEM_CELL_NEIGHBORING_TYPE_H_

#include "config.h"

HBEM_NS_OPEN

/**
 * Different cell neighboring types
 */
enum CellNeighboringType
{
  SamePanel,
  CommonEdge,
  CommonVertex,
  Regular,
  None
};


/**
 * Different methods for detecting cell neighboring types
 */
enum DetectCellNeighboringTypeMethod
{
  SameTriangulations,
  DifferentTriangulations
};


/**
 * Get the string representation of the cell neighboring type.
 *
 * @param s
 * @return
 */
inline const char *
cell_neighboring_type_name(CellNeighboringType n)
{
  switch (n)
    {
      case SamePanel:
        return "same panel";
      case CommonEdge:
        return "common edge";
      case CommonVertex:
        return "common vertex";
      case Regular:
        return "disjoint";
      default:
        return "unknown";
    }
}

HBEM_NS_CLOSE

#endif // HIERBEM_INCLUDE_BEM_CELL_NEIGHBORING_TYPE_H_
