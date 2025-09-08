// Copyright (C) 2020-2025 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your option)
// any later version. The full text of the license can be found in the file
// LICENSE at the top level directory of HierBEM.

Include "CreateAnnulus.geo";

_annulus_inner_radius = 1;
_annulus_outer_radius = 1.5;
_annulus_center_x = 0;
_annulus_center_y = 0;
_annulus_center_z = 0;
_annulus_element_size = 0.3;

//+ Model dimension: 0 for points, 1 for lines and line loops, 2 for surfaces.
_annulus_model_dim = 2;

Call CreateAnnulus;

_annulus_inner_radius = 1;
_annulus_outer_radius = 1.5;
_annulus_center_x = 10;
_annulus_center_y = 10;
_annulus_center_z = 0;
_annulus_element_size = 0.3;

//+ Model dimension: 0 for points, 1 for lines and line loops, 2 for surfaces.
_annulus_model_dim = 1;

Call CreateAnnulus;