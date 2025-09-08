// Copyright (C) 2020-2025 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your option)
// any later version. The full text of the license can be found in the file
// LICENSE at the top level directory of HierBEM.

//+ This script defines a function to create a 2D annulus on the XY plane.

_annulus_inner_radius = 1;
_annulus_outer_radius = 1.5;
_annulus_center_x = 0;
_annulus_center_y = 0;
_annulus_center_z = 0;
_annulus_element_size = 0.1;

//+ Model dimension: 0 for points, 1 for lines and line loops, 2 for surfaces.
_annulus_model_dim = 2;

Function CreateAnnulus

If (_annulus_model_dim >= 0)
  _annulus_start_point_index = newp - 1;

  Point(_annulus_start_point_index + 1) = {_annulus_center_x, _annulus_center_y, _annulus_center_z, _annulus_element_size};
  Point(_annulus_start_point_index + 2) = {_annulus_center_x + _annulus_inner_radius, _annulus_center_y, _annulus_center_z, _annulus_element_size};
  Point(_annulus_start_point_index + 3) = {_annulus_center_x - _annulus_inner_radius, _annulus_center_y, _annulus_center_z, _annulus_element_size};
  Point(_annulus_start_point_index + 4) = {_annulus_center_x + _annulus_outer_radius, _annulus_center_y, _annulus_center_z, _annulus_element_size};
  Point(_annulus_start_point_index + 5) = {_annulus_center_x - _annulus_outer_radius, _annulus_center_y, _annulus_center_z, _annulus_element_size};
EndIf

If (_annulus_model_dim >= 1)
  // N.B. A maximum circle arc with Pi radian can be created by using the Circle command.
  _annulus_start_line_index = newl - 1;
  
  // Create inner circles
  Circle(_annulus_start_line_index + 1) = { _annulus_start_point_index + 2, _annulus_start_point_index + 1, _annulus_start_point_index + 3 };
  Circle(_annulus_start_line_index + 2) = { _annulus_start_point_index + 3, _annulus_start_point_index + 1, _annulus_start_point_index + 2 };

  // Create outer circles
  Circle(_annulus_start_line_index + 3) = { _annulus_start_point_index + 5, _annulus_start_point_index + 1, _annulus_start_point_index + 4 };
  Circle(_annulus_start_line_index + 4) = { _annulus_start_point_index + 4, _annulus_start_point_index + 1, _annulus_start_point_index + 5 };

  _annulus_start_line_loop_index = newll - 1;

  Line Loop(_annulus_start_line_loop_index + 1) = { _annulus_start_line_index + 1, _annulus_start_line_index + 2};
  Line Loop(_annulus_start_line_loop_index + 2) = { _annulus_start_line_index + 3, _annulus_start_line_index + 4};
EndIf

If (_annulus_model_dim >= 2)
  _annulus_start_surface_index = news - 1;

  Plane Surface (_annulus_start_surface_index + 1) =  {_annulus_start_line_loop_index + 1, _annulus_start_line_loop_index + 2};
EndIf

Return
