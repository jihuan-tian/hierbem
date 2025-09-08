// Copyright (C) 2020-2025 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your option)
// any later version. The full text of the license can be found in the file
// LICENSE at the top level directory of HierBEM.

//+ This script defines a function to create a 2D rectangle on the XY plane.

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
//+ Cube model dimension: 0 for points, 1 for lines and line loops, 2 for surfaces.
_rect_model_dim = 2;

Function CreateRect

If (_rect_model_dim >= 0)
  _rect_start_point_index = newp - 1;

  Point (_rect_start_point_index + 1) = {0, 0, 0, _rect_element_size};
  Point (_rect_start_point_index + 2) = {0 + _rect_x_dim, 0, 0, _rect_element_size};
  Point (_rect_start_point_index + 3) = {0 + _rect_x_dim, 0 + _rect_y_dim, 0, _rect_element_size};
  Point (_rect_start_point_index + 4) = {0, 0 + _rect_y_dim, 0, _rect_element_size};

  _rect_point_num = 4;
EndIf

If (_rect_model_dim >= 1)
  _rect_start_line_index = newl - 1;

  Line(_rect_start_line_index + 1) = {_rect_start_point_index + 1, _rect_start_point_index + 2};
  Line(_rect_start_line_index + 2) = {_rect_start_point_index + 2, _rect_start_point_index + 3};
  Line(_rect_start_line_index + 3) = {_rect_start_point_index + 3, _rect_start_point_index + 4};
  Line(_rect_start_line_index + 4) = {_rect_start_point_index + 4, _rect_start_point_index + 1};

  _rect_curve_num = 4;

  _rect_start_line_loop_index = newll - 1;

  Line Loop(_rect_start_line_loop_index + 1) = {_rect_start_line_index + 1, _rect_start_line_index + 4, _rect_start_line_index + 3, _rect_start_line_index + 2};
EndIf

If (_rect_model_dim >= 2)
  _rect_start_surface_index = news - 1;

  Plane Surface(_rect_start_surface_index + 1) = _rect_surface_orient * {_rect_start_line_loop_index + 1};

  _rect_surface_num = 1;
EndIf

// Rotate and translate the created entities according to the specified dimension.
If (_rect_model_dim == 0)
  Rotate {{0, 0, 1}, {0, 0, 0}, _rect_rot_z} {Point {_rect_start_point_index + 1:_rect_start_point_index + _rect_point_num};}
  Translate {_rect_origin_x, _rect_origin_y, _rect_origin_z} {Point {_rect_start_point_index + 1:_rect_start_point_index + _rect_point_num};}
ElseIf (_rect_model_dim == 1)
  Rotate {{0, 0, 1}, {0, 0, 0}, _rect_rot_z} {Point {_rect_start_point_index + 1:_rect_start_point_index + _rect_point_num}; Curve {_rect_start_line_index + 1:_rect_start_line_index + _rect_curve_num};}
  Translate {_rect_origin_x, _rect_origin_y, _rect_origin_z} {Point {_rect_start_point_index + 1:_rect_start_point_index + _rect_point_num}; Curve {_rect_start_line_index + 1:_rect_start_line_index + _rect_curve_num};}
ElseIf (_rect_model_dim == 2)
  Rotate {{0, 0, 1}, {0, 0, 0}, _rect_rot_z} {Point {_rect_start_point_index + 1:_rect_start_point_index + _rect_point_num}; Curve {_rect_start_line_index + 1:_rect_start_line_index + _rect_curve_num}; Surface {_rect_start_surface_index + 1:_rect_start_surface_index + _rect_surface_num};}
  Translate {_rect_origin_x, _rect_origin_y, _rect_origin_z} {Point {_rect_start_point_index + 1:_rect_start_point_index + _rect_point_num}; Curve {_rect_start_line_index + 1:_rect_start_line_index + _rect_curve_num}; Surface {_rect_start_surface_index + 1:_rect_start_surface_index + _rect_surface_num};}
EndIf

Return
