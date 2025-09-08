// Copyright (C) 2020-2025 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your option)
// any later version. The full text of the license can be found in the file
// LICENSE at the top level directory of HierBEM.

Include "CreateCylinder.geo";

_cylinder_radius = 1;
_cylinder_height = 3;
_cylinder_bottom_center_x = 10;
_cylinder_bottom_center_y = 10;
_cylinder_bottom_center_z = -5;
_cylinder_rot_x = -5 * Pi / 180;
_cylinder_rot_y = 10 * Pi / 180;
_cylinder_rot_z = 0;
_cylinder_element_size = 1;
_cylinder_surface_orient = -1;
_cylinder_model_dim = 2;

Call CreateCylinder;

_cylinder_radius = 3;
_cylinder_height = 3;
_cylinder_bottom_center_x = -10;
_cylinder_bottom_center_y = -10;
_cylinder_bottom_center_z = 5;
_cylinder_rot_x = 15 * Pi / 180;
_cylinder_rot_y = -20 * Pi / 180;
_cylinder_rot_z = 0;
_cylinder_element_size = 1;
_cylinder_surface_orient = 1;
_cylinder_model_dim = 3;

Call CreateCylinder;
