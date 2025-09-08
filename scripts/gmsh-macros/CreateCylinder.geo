// Copyright (C) 2020-2025 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your option)
// any later version. The full text of the license can be found in the file
// LICENSE at the top level directory of HierBEM.

//+ This script defines a function to create a cylinder.

_cylinder_radius = 1;
_cylinder_height = 1;
_cylinder_bottom_center_x = 0;
_cylinder_bottom_center_y = 0;
_cylinder_bottom_center_z = 0;
//+ Euler ZYX angles for the cylinder's rotation.
_cylinder_rot_x = 0;
_cylinder_rot_y = 0;
_cylinder_rot_z = Pi/2;
_cylinder_element_size = 0.1;
//+ Surface orientation: -1 for inward, 1 for outward.
_cylinder_surface_orient = 1;
//+ Cylinder model dimension: 0 for points, 1 for lines and line loops, 2 for surfaces and surface loops, 3 for volume.
_cylinder_model_dim = 3;

Function CreateCylinder

If (_cylinder_model_dim >= 0)
  _cylinder_start_point_index = newp - 1;

  //+ Initial circle points of the cylinder bottom.
  Point(_cylinder_start_point_index + 1) = {0, 0, 0, _cylinder_element_size};
  Point(_cylinder_start_point_index + 2) = {_cylinder_radius, 0, 0, _cylinder_element_size};
  Point(_cylinder_start_point_index + 3) = {0, _cylinder_radius, 0, _cylinder_element_size};
  Point(_cylinder_start_point_index + 4) = {-_cylinder_radius, 0, 0, _cylinder_element_size};
  Point(_cylinder_start_point_index + 5) = {0, -_cylinder_radius, 0, _cylinder_element_size};

  //+ Initial circle points of the cylinder top.
  Point(_cylinder_start_point_index + 6) = {0, 0, _cylinder_height, _cylinder_element_size};
  Point(_cylinder_start_point_index + 7) = {_cylinder_radius, 0, _cylinder_height, _cylinder_element_size};
  Point(_cylinder_start_point_index + 8) = {0, _cylinder_radius, _cylinder_height, _cylinder_element_size};
  Point(_cylinder_start_point_index + 9) = {-_cylinder_radius, 0, _cylinder_height, _cylinder_element_size};
  Point(_cylinder_start_point_index + 10) = {0, -_cylinder_radius, _cylinder_height, _cylinder_element_size};

  _cylinder_point_num = 10;
EndIf

If (_cylinder_model_dim >= 1)
  _cylinder_start_line_index = newl - 1;

  //+ Create cylinder bottom circle.
  Circle(_cylinder_start_line_index + 1) = {_cylinder_start_point_index + 2, _cylinder_start_point_index + 1, _cylinder_start_point_index + 3};
  Circle(_cylinder_start_line_index + 2) = {_cylinder_start_point_index + 3, _cylinder_start_point_index + 1, _cylinder_start_point_index + 4};
  Circle(_cylinder_start_line_index + 3) = {_cylinder_start_point_index + 4, _cylinder_start_point_index + 1, _cylinder_start_point_index + 5};
  Circle(_cylinder_start_line_index + 4) = {_cylinder_start_point_index + 5, _cylinder_start_point_index + 1, _cylinder_start_point_index + 2};

  //+ Create cylinder top circle.
  Circle(_cylinder_start_line_index + 5) = {_cylinder_start_point_index + 7, _cylinder_start_point_index + 6, _cylinder_start_point_index + 8};
  Circle(_cylinder_start_line_index + 6) = {_cylinder_start_point_index + 8, _cylinder_start_point_index + 6, _cylinder_start_point_index + 9};
  Circle(_cylinder_start_line_index + 7) = {_cylinder_start_point_index + 9, _cylinder_start_point_index + 6, _cylinder_start_point_index + 10};
  Circle(_cylinder_start_line_index + 8) = {_cylinder_start_point_index + 10, _cylinder_start_point_index + 6, _cylinder_start_point_index + 7};

  //+ Create cylinder side lines.
  Line(_cylinder_start_line_index + 9) = {_cylinder_start_point_index + 2, _cylinder_start_point_index + 7};
  Line(_cylinder_start_line_index + 10) = {_cylinder_start_point_index + 3, _cylinder_start_point_index + 8};
  Line(_cylinder_start_line_index + 11) = {_cylinder_start_point_index + 4, _cylinder_start_point_index + 9};
  Line(_cylinder_start_line_index + 12) = {_cylinder_start_point_index + 5, _cylinder_start_point_index + 10};

  _cylinder_curve_num = 12;

  //+ /////////////////////////////
  //+ Create line loops.
  //+ /////////////////////////////
  _cylinder_start_line_loop_index = newll - 1;

  //+ Create line loop for cylinder bottom circle.
  Line Loop(_cylinder_start_line_loop_index + 1) = {-(_cylinder_start_line_index + 1), -(_cylinder_start_line_index + 4), -(_cylinder_start_line_index + 3), -(_cylinder_start_line_index + 2)};

  //+ Create line loop for cylinder top circle.
  Line Loop(_cylinder_start_line_loop_index + 2) = {_cylinder_start_line_index + 5, _cylinder_start_line_index + 6, _cylinder_start_line_index + 7, _cylinder_start_line_index + 8};

  //+ Create line loop for the 1st side line loop.
  Line Loop(_cylinder_start_line_loop_index + 3) = {_cylinder_start_line_index + 1, _cylinder_start_line_index + 10, -(_cylinder_start_line_index + 5), -(_cylinder_start_line_index + 9)};

  //+ Create line loop for the 2nd side line loop.
  Line Loop(_cylinder_start_line_loop_index + 4) = {_cylinder_start_line_index + 2, _cylinder_start_line_index + 11, -(_cylinder_start_line_index + 6), -(_cylinder_start_line_index + 10)};

  //+ Create line loop for the 3rd side line loop.
  Line Loop(_cylinder_start_line_loop_index + 5) = {_cylinder_start_line_index + 3, _cylinder_start_line_index + 12, -(_cylinder_start_line_index + 7), -(_cylinder_start_line_index + 11)};

  //+ Create line loop for the 4th side line loop.
  Line Loop(_cylinder_start_line_loop_index + 6) = {_cylinder_start_line_index + 4, _cylinder_start_line_index + 9, -(_cylinder_start_line_index + 8), -(_cylinder_start_line_index + 12)};
EndIf

If (_cylinder_model_dim >= 2)
  _cylinder_start_surface_index = news - 1;

  //+ Create cylinder bottom circle surface.
  Plane Surface(_cylinder_start_surface_index + 1) = _cylinder_surface_orient * {_cylinder_start_line_loop_index + 1};

  //+ Create cylinder top circle surface.
  Plane Surface(_cylinder_start_surface_index + 2) = _cylinder_surface_orient * {_cylinder_start_line_loop_index + 2};

  //+ Create four cylinder side surfaces.
  //+ N.B. In Gmsh 4.3.0, "Surface" replaces the deprecated "Ruled Surface" command.
  For _cylinder_i In {1:4}
    Ruled Surface(_cylinder_start_surface_index + 2 + _cylinder_i) = _cylinder_surface_orient * {_cylinder_start_line_loop_index + 2 + _cylinder_i};
  EndFor

  _cylinder_surface_num = 6;

  //+ ///////////////////////////////
  //+ Create surface loop.
  //+ ///////////////////////////////
  _cylinder_start_surface_loop_index = newsl - 1;
  Surface Loop(_cylinder_start_surface_loop_index + 1) = {_cylinder_start_surface_index + 1:_cylinder_start_surface_index + 6};
EndIf

If (_cylinder_model_dim == 3)
  _cylinder_start_volume_index = newv - 1;
  Volume(_cylinder_start_volume_index + 1) = {_cylinder_start_surface_loop_index + 1};
  _cylinder_volume_num = 1;
EndIf

// Rotate and translate the created entities according to the specified dimension.
If (_cylinder_model_dim == 0)
  Rotate {{0, 0, 1}, {0, 0, 0}, _cylinder_rot_z} {Point {_cylinder_start_point_index + 1:_cylinder_start_point_index + _cylinder_point_num};}
  Rotate {{0, 1, 0}, {0, 0, 0}, _cylinder_rot_y} {Point {_cylinder_start_point_index + 1:_cylinder_start_point_index + _cylinder_point_num};}
  Rotate {{1, 0, 0}, {0, 0, 0}, _cylinder_rot_x} {Point {_cylinder_start_point_index + 1:_cylinder_start_point_index + _cylinder_point_num};}
  Translate {_cylinder_bottom_center_x, _cylinder_bottom_center_y, _cylinder_bottom_center_z} {Point {_cylinder_start_point_index + 1:_cylinder_start_point_index + _cylinder_point_num};}
ElseIf (_cylinder_model_dim == 1)
  Rotate {{0, 0, 1}, {0, 0, 0}, _cylinder_rot_z} {Point {_cylinder_start_point_index + 1:_cylinder_start_point_index + _cylinder_point_num}; Curve {_cylinder_start_line_index + 1:_cylinder_start_line_index + _cylinder_curve_num};}
  Rotate {{0, 1, 0}, {0, 0, 0}, _cylinder_rot_y} {Point {_cylinder_start_point_index + 1:_cylinder_start_point_index + _cylinder_point_num}; Curve {_cylinder_start_line_index + 1:_cylinder_start_line_index + _cylinder_curve_num};}
  Rotate {{1, 0, 0}, {0, 0, 0}, _cylinder_rot_x} {Point {_cylinder_start_point_index + 1:_cylinder_start_point_index + _cylinder_point_num}; Curve {_cylinder_start_line_index + 1:_cylinder_start_line_index + _cylinder_curve_num};}
  Translate {_cylinder_bottom_center_x, _cylinder_bottom_center_y, _cylinder_bottom_center_z} {Point {_cylinder_start_point_index + 1:_cylinder_start_point_index + _cylinder_point_num}; Curve {_cylinder_start_line_index + 1:_cylinder_start_line_index + _cylinder_curve_num};}
ElseIf (_cylinder_model_dim == 2)
  Rotate {{0, 0, 1}, {0, 0, 0}, _cylinder_rot_z} {Point {_cylinder_start_point_index + 1:_cylinder_start_point_index + _cylinder_point_num}; Curve {_cylinder_start_line_index + 1:_cylinder_start_line_index + _cylinder_curve_num}; Surface {_cylinder_start_surface_index + 1:_cylinder_start_surface_index + _cylinder_surface_num};}
  Rotate {{0, 1, 0}, {0, 0, 0}, _cylinder_rot_y} {Point {_cylinder_start_point_index + 1:_cylinder_start_point_index + _cylinder_point_num}; Curve {_cylinder_start_line_index + 1:_cylinder_start_line_index + _cylinder_curve_num}; Surface {_cylinder_start_surface_index + 1:_cylinder_start_surface_index + _cylinder_surface_num};}
  Rotate {{1, 0, 0}, {0, 0, 0}, _cylinder_rot_x} {Point {_cylinder_start_point_index + 1:_cylinder_start_point_index + _cylinder_point_num}; Curve {_cylinder_start_line_index + 1:_cylinder_start_line_index + _cylinder_curve_num}; Surface {_cylinder_start_surface_index + 1:_cylinder_start_surface_index + _cylinder_surface_num};}
  Translate {_cylinder_bottom_center_x, _cylinder_bottom_center_y, _cylinder_bottom_center_z} {Point {_cylinder_start_point_index + 1:_cylinder_start_point_index + _cylinder_point_num}; Curve {_cylinder_start_line_index + 1:_cylinder_start_line_index + _cylinder_curve_num}; Surface {_cylinder_start_surface_index + 1:_cylinder_start_surface_index + _cylinder_surface_num};}
ElseIf (_cylinder_model_dim == 3)
  Rotate {{0, 0, 1}, {0, 0, 0}, _cylinder_rot_z} {Point {_cylinder_start_point_index + 1:_cylinder_start_point_index + _cylinder_point_num}; Curve {_cylinder_start_line_index + 1:_cylinder_start_line_index + _cylinder_curve_num}; Surface {_cylinder_start_surface_index + 1:_cylinder_start_surface_index + _cylinder_surface_num}; Volume {_cylinder_start_volume_index + 1:_cylinder_start_volume_index + _cylinder_volume_num};}
  Rotate {{0, 1, 0}, {0, 0, 0}, _cylinder_rot_y} {Point {_cylinder_start_point_index + 1:_cylinder_start_point_index + _cylinder_point_num}; Curve {_cylinder_start_line_index + 1:_cylinder_start_line_index + _cylinder_curve_num}; Surface {_cylinder_start_surface_index + 1:_cylinder_start_surface_index + _cylinder_surface_num}; Volume {_cylinder_start_volume_index + 1:_cylinder_start_volume_index + _cylinder_volume_num};}
  Rotate {{1, 0, 0}, {0, 0, 0}, _cylinder_rot_x} {Point {_cylinder_start_point_index + 1:_cylinder_start_point_index + _cylinder_point_num}; Curve {_cylinder_start_line_index + 1:_cylinder_start_line_index + _cylinder_curve_num}; Surface {_cylinder_start_surface_index + 1:_cylinder_start_surface_index + _cylinder_surface_num}; Volume {_cylinder_start_volume_index + 1:_cylinder_start_volume_index + _cylinder_volume_num};}
  Translate {_cylinder_bottom_center_x, _cylinder_bottom_center_y, _cylinder_bottom_center_z} {Point {_cylinder_start_point_index + 1:_cylinder_start_point_index + _cylinder_point_num}; Curve {_cylinder_start_line_index + 1:_cylinder_start_line_index + _cylinder_curve_num}; Surface {_cylinder_start_surface_index + 1:_cylinder_start_surface_index + _cylinder_surface_num}; Volume {_cylinder_start_volume_index + 1:_cylinder_start_volume_index + _cylinder_volume_num};}
EndIf

Return
