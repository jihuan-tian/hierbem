// Copyright (C) 2020-2025 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your option)
// any later version. The full text of the license can be found in the file
// LICENSE at the top level directory of HierBEM.

Include "CreateRect.geo";

_rect_origin_x = 0;
_rect_origin_y = 0;
_rect_origin_z = 0;
_rect_x_dim = 1;
_rect_y_dim = 1;
_rect_element_size = 0.1;
//+ Euler Z angles for the cube's rotation.
_rect_rot_z = 0;
//+ Surface orientation: -1 for inward, 1 for outward.
_rect_surface_orient = 1;
//+ Cube model dimension: 0 for points, 1 for lines and line loops, 2 for surfaces and surface loops.
_rect_model_dim = 0;

Call CreateRect;

_rect_origin_x = 10;
_rect_origin_y = 10;
_rect_origin_z = 0;
_rect_x_dim = 1;
_rect_y_dim = 1;
_rect_element_size = 0.1;
//+ Euler Z angles for the cube's rotation.
_rect_rot_z = 30 * Pi / 180.0;
//+ Surface orientation: -1 for inward, 1 for outward.
_rect_surface_orient = 1;
//+ Cube model dimension: 0 for points, 1 for lines and line loops, 2 for surfaces and surface loops.
_rect_model_dim = 1;

Call CreateRect;

_rect_origin_x = -10;
_rect_origin_y = -10;
_rect_origin_z = 0;
_rect_x_dim = 1;
_rect_y_dim = 1;
_rect_element_size = 0.1;
//+ Euler Z angles for the cube's rotation.
_rect_rot_z = 130 * Pi / 180.0;
//+ Surface orientation: -1 for inward, 1 for outward.
_rect_surface_orient = 1;
//+ Cube model dimension: 0 for points, 1 for lines and line loops, 2 for surfaces and surface loops.
_rect_model_dim = 2;

Call CreateRect;