// Copyright (C) 2023 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

Mesh.Algorithm = 6;

Include "../../scripts/gmsh-macros/CreateSphere.geo";

_sphere_radius = 1;
_sphere_center_x = 0;
_sphere_center_y = 0;
_sphere_center_z = 0;
_sphere_surface_orient = -1;
_sphere_element_size = 0.8;
//+ Sphere model dimension: 0 for points, 1 for lines and line loops, 2 for surfaces and surface loops, 3 for volume.
_sphere_model_dim = 3;

Call CreateSphere;

//+ Create a physical group for the surfaces in the upper hemisphere.
Physical Surface(1) = {21, 22, 23, 24};
//+ Create a physical group for the surfaces in the lower hemisphere.
Physical Surface(2) = {25, 26, 28, 27};

// Create a physical group for the sphere volume.
Physical Volume(1) = {30};
