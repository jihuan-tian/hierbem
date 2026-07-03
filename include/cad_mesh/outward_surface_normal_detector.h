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
 * @file outward_surface_normal_detector.h
 * @brief Definition of class @p OutwardSurfaceNormalDetector.
 * @ingroup
 *
 * @date 2026-05-29
 * @author Jihuan Tian
 */

#ifndef HIERBEM_INCLUDE_CAD_MESH_OUTWARD_SURFACE_NORMAL_DETECTOR_H_
#define HIERBEM_INCLUDE_CAD_MESH_OUTWARD_SURFACE_NORMAL_DETECTOR_H_

#include "config.h"

HBEM_NS_OPEN

/**
 * Surface normal vector direction detector which always returns false about the
 * normal vector's inwardness with respect to the associated volume. This means
 * the normal vector points outside the volume.
 */
class OutwardSurfaceNormalDetector
{
public:
  bool
  is_normal_vector_inward([[maybe_unused]] const types::material_id m) const
  {
    return false;
  }
};

HBEM_NS_CLOSE

#endif // HIERBEM_INCLUDE_CAD_MESH_OUTWARD_SURFACE_NORMAL_DETECTOR_H_
