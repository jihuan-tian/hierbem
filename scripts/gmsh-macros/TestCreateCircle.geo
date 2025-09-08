// Copyright (C) 2020-2025 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your option)
// any later version. The full text of the license can be found in the file
// LICENSE at the top level directory of HierBEM.

Include "CreateCircle.geo";

_circle_radius = 1;
_circle_center_x = 5;
_circle_center_y = 0;
_circle_center_z = 0;
_circle_surface_orient = 1;
_circle_element_size = 1;
//+ Sphere model dimension: 0 for points, 1 for lines and line loops, 2 for surfaces.
_circle_model_dim = 0;

Call CreateCircle;

_circle_radius = 1;
_circle_center_x = -5;
_circle_center_y = 0;
_circle_center_z = 0;
_circle_surface_orient = 1;
_circle_element_size = 1;
//+ Sphere model dimension: 0 for points, 1 for lines and line loops, 2 for surfaces.
_circle_model_dim = 1;

Call CreateCircle;

_circle_radius = 2;
_circle_center_x = 0;
_circle_center_y = 5;
_circle_center_z = 0;
_circle_surface_orient = 1;
_circle_element_size = 1;
//+ Sphere model dimension: 0 for points, 1 for lines and line loops, 2 for surfaces.
_circle_model_dim = 2;

Call CreateCircle;

_circle_radius = 2;
_circle_center_x = 0;
_circle_center_y = -5;
_circle_center_z = 0;
_circle_surface_orient = 1;
_circle_element_size = 1;
//+ Sphere model dimension: 0 for points, 1 for lines and line loops, 2 for surfaces.
_circle_model_dim = 2;

Call CreateCircle;
