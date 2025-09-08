// Copyright (C) 2020-2025 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your option)
// any later version. The full text of the license can be found in the file
// LICENSE at the top level directory of HierBEM.

// Initialize parameters required for creating a sphere.
_sphere_radius = 1;
_sphere_center_x = 0;
_sphere_center_y = 0;
_sphere_center_z = 0;
_sphere_surface_orient = 1;
_sphere_element_size = 0.1;
//+ Sphere model dimension: 0 for points, 1 for lines and line loops, 2 for surfaces and surface loops, 3 for volume.
_sphere_model_dim = 3;

Function CreateSphere

// The sphere model will be created from a lower dimension to a higher
// dimension, i.e. in the order of vertices, lines, surfaces and
// volumes.
// 
// The vertices are distributed as below.
//
//                2       7
//               z|     /
//                |   /
//                | /
//       5--------1--------4
//               /|       y
//             /  |
//         x /    |
//          6     3
//
//
// Then a series quarter circles will be generated according to the
// following vertex sequences:
// Circle 1 = [2, 1, 4]
// Circle 2 = [2, 1, 5]
// Circle 3 = [2, 1, 6]
// Circle 4 = [2, 1, 7]
// Circle 5 = [3, 1, 4]
// Circle 6 = [3, 1, 5]
// Circle 7 = [3, 1, 6]
// Circle 8 = [3, 1, 7]
// Circle 9 = [6, 1, 4]
// Circle 10 = [4, 1, 7]
// Circle 11 = [7, 1, 5]
// Circle 12 = [5, 1, 6]
//
// Next, line loops will be created from the above quarter circles as below.
// Line loop 1 = [4, 11, -2]
//
//
//
// 

If (_sphere_model_dim >= 0)
  _sphere_start_point_index = newp - 1;

  // Origin
  Point(_sphere_start_point_index + 1) = {_sphere_center_x, _sphere_center_y, _sphere_center_z, _sphere_element_size};
  // Top
  Point(_sphere_start_point_index + 2) = {_sphere_center_x, _sphere_center_y, _sphere_center_z + _sphere_radius, _sphere_element_size};
  // Bottom
  Point(_sphere_start_point_index + 3) = {_sphere_center_x, _sphere_center_y, _sphere_center_z - _sphere_radius, _sphere_element_size};
  // Right
  Point(_sphere_start_point_index + 4) = {_sphere_center_x, _sphere_center_y + _sphere_radius, _sphere_center_z, _sphere_element_size};
  // Left
  Point(_sphere_start_point_index + 5) = {_sphere_center_x, _sphere_center_y - _sphere_radius, _sphere_center_z, _sphere_element_size};
  // Front
  Point(_sphere_start_point_index + 6) = {_sphere_center_x + _sphere_radius, _sphere_center_y, _sphere_center_z, _sphere_element_size};
  // Back
  Point(_sphere_start_point_index + 7) = {_sphere_center_x - _sphere_radius, _sphere_center_y, _sphere_center_z, _sphere_element_size};
EndIf

If (_sphere_model_dim >= 1)
  // Create circles
  // N.B. A maximum circle arc with Pi radian can be created by using the Circle command.
  _sphere_start_line_index = newl - 1;
  Circle(_sphere_start_line_index + 1) = { _sphere_start_point_index + 2, _sphere_start_point_index + 1, _sphere_start_point_index + 4 };
  Circle(_sphere_start_line_index + 2) = { _sphere_start_point_index + 2, _sphere_start_point_index + 1, _sphere_start_point_index + 5 };
  Circle(_sphere_start_line_index + 3) = { _sphere_start_point_index + 2, _sphere_start_point_index + 1, _sphere_start_point_index + 6 };
  Circle(_sphere_start_line_index + 4) = { _sphere_start_point_index + 2, _sphere_start_point_index + 1, _sphere_start_point_index + 7 };
  Circle(_sphere_start_line_index + 5) = { _sphere_start_point_index + 3, _sphere_start_point_index + 1, _sphere_start_point_index + 4 };
  Circle(_sphere_start_line_index + 6) = { _sphere_start_point_index + 3, _sphere_start_point_index + 1, _sphere_start_point_index + 5 };
  Circle(_sphere_start_line_index + 7) = { _sphere_start_point_index + 3, _sphere_start_point_index + 1, _sphere_start_point_index + 6 };
  Circle(_sphere_start_line_index + 8) = { _sphere_start_point_index + 3, _sphere_start_point_index + 1, _sphere_start_point_index + 7 };
  Circle(_sphere_start_line_index + 9) = { _sphere_start_point_index + 6, _sphere_start_point_index + 1, _sphere_start_point_index + 4 };
  Circle(_sphere_start_line_index + 10) = { _sphere_start_point_index + 4, _sphere_start_point_index + 1, _sphere_start_point_index + 7 };
  Circle(_sphere_start_line_index + 11) = { _sphere_start_point_index + 7, _sphere_start_point_index + 1, _sphere_start_point_index + 5 };
  Circle(_sphere_start_line_index + 12) = { _sphere_start_point_index + 5, _sphere_start_point_index + 1, _sphere_start_point_index + 6 };

  _sphere_start_line_loop_index = newll - 1;

  Line Loop(_sphere_start_line_loop_index + 1) = { _sphere_start_line_index + 4, _sphere_start_line_index + 11, -(_sphere_start_line_index + 2) };
  Line Loop(_sphere_start_line_loop_index + 2) = { _sphere_start_line_index + 2, _sphere_start_line_index + 12, -(_sphere_start_line_index + 3) };
  Line Loop(_sphere_start_line_loop_index + 3) = { _sphere_start_line_index + 3, _sphere_start_line_index + 9, -(_sphere_start_line_index + 1) };
  Line Loop(_sphere_start_line_loop_index + 4) = { _sphere_start_line_index + 1, _sphere_start_line_index + 10, -(_sphere_start_line_index + 4) };
  Line Loop(_sphere_start_line_loop_index + 5) = { _sphere_start_line_index + 8, _sphere_start_line_index + 11, -(_sphere_start_line_index + 6) };
  Line Loop(_sphere_start_line_loop_index + 6) = { _sphere_start_line_index + 6, _sphere_start_line_index + 12, -(_sphere_start_line_index + 7) };
  Line Loop(_sphere_start_line_loop_index + 7) = { _sphere_start_line_index + 7, _sphere_start_line_index + 9, -(_sphere_start_line_index + 5) };
  Line Loop(_sphere_start_line_loop_index + 8) = { _sphere_start_line_index + 5, _sphere_start_line_index + 10, -(_sphere_start_line_index + 8) };
EndIf

If (_sphere_model_dim >= 2)
  _sphere_start_surface_index = news - 1;

  For _sphere_i In {1:4}
    Surface(_sphere_start_surface_index + _sphere_i) = _sphere_surface_orient * { _sphere_start_line_loop_index + _sphere_i} In Sphere { _sphere_start_point_index + 1};
  EndFor

  For _sphere_i In {5:8}
    Surface(_sphere_start_surface_index + _sphere_i) = _sphere_surface_orient * { -(_sphere_start_line_loop_index + _sphere_i)} In Sphere { _sphere_start_point_index + 1};
  EndFor

  _sphere_surface_loop_index = newsl;
  Surface Loop(_sphere_surface_loop_index) = {(_sphere_start_surface_index + 1):(_sphere_start_surface_index + 8)};

  // All the sphere surface should be grouped into a physical surface,
  // then the generated mesh can be read by deal.ii.
  // 2022-11-16: comment out this line to manually create physical surfaces.
  // Physical Surface(_sphere_surface_loop_index) = {(_sphere_start_surface_index + 1):(_sphere_start_surface_index + 8)};
EndIf

If (_sphere_model_dim == 3)
  _sphere_start_volume_index = newv - 1;
  Volume(_sphere_start_volume_index + 1) = {_sphere_surface_loop_index};
EndIf

Return
