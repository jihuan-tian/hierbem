// Copyright (C) 2020-2025 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your option)
// any later version. The full text of the license can be found in the file
// LICENSE at the top level directory of HierBEM.

//+ This script defines a function to create a 2D circle on the XY plane.

_circle_radius = 1;
_circle_center_x = 0;
_circle_center_y = 0;
_circle_center_z = 0;
_circle_surface_orient = 1;
_circle_element_size = 0.1;
//+ Sphere model dimension: 0 for points, 1 for lines and line loops, 2 for surfaces.
_circle_model_dim = 2;

Function CreateCircle

If (_circle_model_dim >= 0)
  _circle_start_point_index = newp - 1;

  Point(_circle_start_point_index + 1) = {_circle_center_x, _circle_center_y, _circle_center_z, _circle_element_size};
  Point(_circle_start_point_index + 2) = {_circle_center_x + _circle_radius, _circle_center_y, _circle_center_z, _circle_element_size};
  Point(_circle_start_point_index + 3) = {_circle_center_x - _circle_radius, _circle_center_y, _circle_center_z, _circle_element_size};
EndIf

If (_circle_model_dim >= 1)
  // Create circles
  // N.B. A maximum circle arc with Pi radian can be created by using the Circle command.
  _circle_start_line_index = newl - 1;
  
  Circle(_circle_start_line_index + 1) = { _circle_start_point_index + 2, _circle_start_point_index + 1, _circle_start_point_index + 3 };
  Circle(_circle_start_line_index + 2) = { _circle_start_point_index + 3, _circle_start_point_index + 1, _circle_start_point_index + 2 };

  _circle_start_line_loop_index = newll - 1;

  Line Loop(_circle_start_line_loop_index + 1) = { _circle_start_line_index + 1, _circle_start_line_index + 2};
EndIf

If (_circle_model_dim >= 2)
  _circle_start_surface_index = news - 1;

  Plane Surface (_circle_start_surface_index + 1) = _circle_surface_orient * { _circle_start_line_loop_index + 1};
EndIf

Return
