// Copyright (C) 2020-2025 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your option)
// any later version. The full text of the license can be found in the file
// LICENSE at the top level directory of HierBEM.

//+ This script defines a function to create a cube.

_cube_origin_x = 0;
_cube_origin_y = 0;
_cube_origin_z = 0;
_cube_x_dim = 1;
_cube_y_dim = 1;
_cube_z_dim = 1;
_cube_element_size = 0.1;
//+ Euler ZYX angles for the cube's rotation.
_cube_rot_x = 0;
_cube_rot_y = 0;
_cube_rot_z = 0;
//+ Surface orientation: -1 for inward, 1 for outward.
_cube_surface_orient = 1;
//+ Cube model dimension: 0 for points, 1 for lines and line loops, 2 for surfaces and surface loops, 3 for volume.
_cube_model_dim = 3;

Function CreateCube

If (_cube_model_dim >= 0)
  _cube_start_point_index = newp - 1;

  Point (_cube_start_point_index + 1) = {0, 0, 0, _cube_element_size};
  Point (_cube_start_point_index + 2) = {0 + _cube_x_dim, 0, 0, _cube_element_size};
  Point (_cube_start_point_index + 3) = {0 + _cube_x_dim, 0 + _cube_y_dim, 0, _cube_element_size};
  Point (_cube_start_point_index + 4) = {0, 0 + _cube_y_dim, 0, _cube_element_size};
  Point (_cube_start_point_index + 5) = {0, 0, 0 + _cube_z_dim, _cube_element_size};
  Point (_cube_start_point_index + 6) = {0 + _cube_x_dim, 0, 0 + _cube_z_dim, _cube_element_size};
  Point (_cube_start_point_index + 7) = {0 + _cube_x_dim, 0 + _cube_y_dim, 0 + _cube_z_dim, _cube_element_size};
  Point (_cube_start_point_index + 8) = {0, 0 + _cube_y_dim, 0 + _cube_z_dim, _cube_element_size};

  _cube_point_num = 8;
EndIf

If (_cube_model_dim >= 1)
  _cube_start_line_index = newl - 1;

  Line(_cube_start_line_index + 1) = {_cube_start_point_index + 1, _cube_start_point_index + 2};
  Line(_cube_start_line_index + 2) = {_cube_start_point_index + 2, _cube_start_point_index + 3};
  Line(_cube_start_line_index + 3) = {_cube_start_point_index + 3, _cube_start_point_index + 4};
  Line(_cube_start_line_index + 4) = {_cube_start_point_index + 1, _cube_start_point_index + 4};
  Line(_cube_start_line_index + 5) = {_cube_start_point_index + 5, _cube_start_point_index + 6};
  Line(_cube_start_line_index + 6) = {_cube_start_point_index + 6, _cube_start_point_index + 7};
  Line(_cube_start_line_index + 7) = {_cube_start_point_index + 7, _cube_start_point_index + 8};
  Line(_cube_start_line_index + 8) = {_cube_start_point_index + 5, _cube_start_point_index + 8};
  Line(_cube_start_line_index + 9) = {_cube_start_point_index + 1, _cube_start_point_index + 5};
  Line(_cube_start_line_index + 10) = {_cube_start_point_index + 2, _cube_start_point_index + 6};
  Line(_cube_start_line_index + 11) = {_cube_start_point_index + 3, _cube_start_point_index + 7};
  Line(_cube_start_line_index + 12) = {_cube_start_point_index + 4, _cube_start_point_index + 8};

  _cube_curve_num = 8;

  _cube_start_line_loop_index = newll - 1;

  Line Loop(_cube_start_line_loop_index + 1) = {-(_cube_start_line_index + 1), _cube_start_line_index + 4, -(_cube_start_line_index + 3), -(_cube_start_line_index + 2)};
  Line Loop(_cube_start_line_loop_index + 2) = {_cube_start_line_index + 5, _cube_start_line_index + 6, _cube_start_line_index + 7, -(_cube_start_line_index + 8)};
  Line Loop(_cube_start_line_loop_index + 3) = {-(_cube_start_line_index + 6), -(_cube_start_line_index + 10), _cube_start_line_index + 2, _cube_start_line_index + 11};
  Line Loop(_cube_start_line_loop_index + 4) = {_cube_start_line_index + 8, -(_cube_start_line_index + 12), -(_cube_start_line_index + 4), _cube_start_line_index + 9};
  Line Loop(_cube_start_line_loop_index + 5) = {-(_cube_start_line_index + 5), -(_cube_start_line_index + 9), _cube_start_line_index + 1, _cube_start_line_index + 10};
  Line Loop(_cube_start_line_loop_index + 6) = {-(_cube_start_line_index + 7), -(_cube_start_line_index + 11), _cube_start_line_index + 3, _cube_start_line_index + 12};
EndIf

If (_cube_model_dim >= 2)
  _cube_start_surface_index = news - 1;

  For _cube_i In {1:6}
    Plane Surface(_cube_start_surface_index + _cube_i) = _cube_surface_orient * {_cube_start_line_loop_index + _cube_i};
  EndFor

  _cube_surface_num = 6;

  _cube_surface_loop_index = newsl;

  Surface Loop(_cube_surface_loop_index) = {_cube_start_surface_index + 1:_cube_start_surface_index + 6};
EndIf

If (_cube_model_dim == 3)
  _cube_start_volume_index = newv - 1;
  Volume(_cube_start_volume_index + 1) = {_cube_surface_loop_index};
  _cube_volume_num = 1;
EndIf

// Rotate and translate the created entities according to the specified dimension.
If (_cube_model_dim == 0)
  Rotate {{0, 0, 1}, {0, 0, 0}, _cube_rot_z} {Point {_cube_start_point_index + 1:_cube_start_point_index + _cube_point_num};}
  Rotate {{0, 1, 0}, {0, 0, 0}, _cube_rot_y} {Point {_cube_start_point_index + 1:_cube_start_point_index + _cube_point_num};}
  Rotate {{1, 0, 0}, {0, 0, 0}, _cube_rot_x} {Point {_cube_start_point_index + 1:_cube_start_point_index + _cube_point_num};}
  Translate {_cube_origin_x, _cube_origin_y, _cube_origin_z} {Point {_cube_start_point_index + 1:_cube_start_point_index + _cube_point_num};}
ElseIf (_cube_model_dim == 1)
  Rotate {{0, 0, 1}, {0, 0, 0}, _cube_rot_z} {Point {_cube_start_point_index + 1:_cube_start_point_index + _cube_point_num}; Curve {_cube_start_line_index + 1:_cube_start_line_index + _cube_curve_num};}
  Rotate {{0, 1, 0}, {0, 0, 0}, _cube_rot_y} {Point {_cube_start_point_index + 1:_cube_start_point_index + _cube_point_num}; Curve {_cube_start_line_index + 1:_cube_start_line_index + _cube_curve_num};}
  Rotate {{1, 0, 0}, {0, 0, 0}, _cube_rot_x} {Point {_cube_start_point_index + 1:_cube_start_point_index + _cube_point_num}; Curve {_cube_start_line_index + 1:_cube_start_line_index + _cube_curve_num};}
  Translate {_cube_origin_x, _cube_origin_y, _cube_origin_z} {Point {_cube_start_point_index + 1:_cube_start_point_index + _cube_point_num}; Curve {_cube_start_line_index + 1:_cube_start_line_index + _cube_curve_num};}
ElseIf (_cube_model_dim == 2)
  Rotate {{0, 0, 1}, {0, 0, 0}, _cube_rot_z} {Point {_cube_start_point_index + 1:_cube_start_point_index + _cube_point_num}; Curve {_cube_start_line_index + 1:_cube_start_line_index + _cube_curve_num}; Surface {_cube_start_surface_index + 1:_cube_start_surface_index + _cube_surface_num};}
  Rotate {{0, 1, 0}, {0, 0, 0}, _cube_rot_y} {Point {_cube_start_point_index + 1:_cube_start_point_index + _cube_point_num}; Curve {_cube_start_line_index + 1:_cube_start_line_index + _cube_curve_num}; Surface {_cube_start_surface_index + 1:_cube_start_surface_index + _cube_surface_num};}
  Rotate {{1, 0, 0}, {0, 0, 0}, _cube_rot_x} {Point {_cube_start_point_index + 1:_cube_start_point_index + _cube_point_num}; Curve {_cube_start_line_index + 1:_cube_start_line_index + _cube_curve_num}; Surface {_cube_start_surface_index + 1:_cube_start_surface_index + _cube_surface_num};}
  Translate {_cube_origin_x, _cube_origin_y, _cube_origin_z} {Point {_cube_start_point_index + 1:_cube_start_point_index + _cube_point_num}; Curve {_cube_start_line_index + 1:_cube_start_line_index + _cube_curve_num}; Surface {_cube_start_surface_index + 1:_cube_start_surface_index + _cube_surface_num};}
ElseIf (_cube_model_dim == 3)
  Rotate {{0, 0, 1}, {0, 0, 0}, _cube_rot_z} {Point {_cube_start_point_index + 1:_cube_start_point_index + _cube_point_num}; Curve {_cube_start_line_index + 1:_cube_start_line_index + _cube_curve_num}; Surface {_cube_start_surface_index + 1:_cube_start_surface_index + _cube_surface_num}; Volume {_cube_start_volume_index + 1:_cube_start_volume_index + _cube_volume_num};}
  Rotate {{0, 1, 0}, {0, 0, 0}, _cube_rot_y} {Point {_cube_start_point_index + 1:_cube_start_point_index + _cube_point_num}; Curve {_cube_start_line_index + 1:_cube_start_line_index + _cube_curve_num}; Surface {_cube_start_surface_index + 1:_cube_start_surface_index + _cube_surface_num}; Volume {_cube_start_volume_index + 1:_cube_start_volume_index + _cube_volume_num};}
  Rotate {{1, 0, 0}, {0, 0, 0}, _cube_rot_x} {Point {_cube_start_point_index + 1:_cube_start_point_index + _cube_point_num}; Curve {_cube_start_line_index + 1:_cube_start_line_index + _cube_curve_num}; Surface {_cube_start_surface_index + 1:_cube_start_surface_index + _cube_surface_num}; Volume {_cube_start_volume_index + 1:_cube_start_volume_index + _cube_volume_num};}
  Translate {_cube_origin_x, _cube_origin_y, _cube_origin_z} {Point {_cube_start_point_index + 1:_cube_start_point_index + _cube_point_num}; Curve {_cube_start_line_index + 1:_cube_start_line_index + _cube_curve_num}; Surface {_cube_start_surface_index + 1:_cube_start_surface_index + _cube_surface_num}; Volume {_cube_start_volume_index + 1:_cube_start_volume_index + _cube_volume_num};}
EndIf

Return
