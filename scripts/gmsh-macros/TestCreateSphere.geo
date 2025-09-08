// Copyright (C) 2020-2025 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your option)
// any later version. The full text of the license can be found in the file
// LICENSE at the top level directory of HierBEM.

Include "CreateSphere.geo";

_sphere_radius = 1;
_sphere_center_x = 0;
_sphere_center_y = 0;
_sphere_center_z = 0;
_sphere_element_size = 0.5;
_sphere_surface_orient = 1;
_sphere_model_dim = 0;
Call CreateSphere;

_sphere_radius = 1;
_sphere_center_x = 4;
_sphere_center_y = 0;
_sphere_center_z = 0;
_sphere_element_size = 0.5;
_sphere_surface_orient = 1;
_sphere_model_dim = 1;
Call CreateSphere;

_sphere_radius = 1;
_sphere_center_x = 4;
_sphere_center_y = 4;
_sphere_center_z = 0;
_sphere_element_size = 0.5;
_sphere_surface_orient = 1;
_sphere_model_dim = 2;
Call CreateSphere;

_sphere_radius = 1;
_sphere_center_x = 0;
_sphere_center_y = 4;
_sphere_center_z = 0;
_sphere_element_size = 0.5;
_sphere_surface_orient = 1;
Call CreateSphere;

_sphere_radius = 1;
_sphere_center_x = 2;
_sphere_center_y = 2;
_sphere_center_z = 2;
_sphere_element_size = 0.5;
_sphere_surface_orient = 1;
_sphere_model_dim = 3;
Call CreateSphere;

_sphere_radius = 1;
_sphere_center_x = 2;
_sphere_center_y = 2;
_sphere_center_z = -2;
_sphere_element_size = 0.5;
_sphere_surface_orient = 1;
_sphere_model_dim = 3;
Call CreateSphere;
