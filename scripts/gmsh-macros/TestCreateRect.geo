// Copyright (C) 2020 Jihuan Tian <jihuan_tian@hotmail.com>
//  
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//  
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//  
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
//  

Include "CreateRect.geo";

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
//+ Cube model dimension: 0 for points, 1 for lines and line loops, 2 for surfaces and surface loops.
_rect_model_dim = 0;

Call CreateRect;

_rect_origin_x = 10;
_rect_origin_y = 10;
_rect_origin_z = 0;
_rect_x_dim = 1;
_rect_y_dim = 1;
_rect_element_size = 0.1;
//+ Euler Z angles for the cube's rotation.
_rect_rot_z = 30 * Pi / 180.0;
//+ Surface orientation: -1 for inward, 1 for outward.
_rect_surface_orient = 1;
//+ Cube model dimension: 0 for points, 1 for lines and line loops, 2 for surfaces and surface loops.
_rect_model_dim = 1;

Call CreateRect;

_rect_origin_x = -10;
_rect_origin_y = -10;
_rect_origin_z = 0;
_rect_x_dim = 1;
_rect_y_dim = 1;
_rect_element_size = 0.1;
//+ Euler Z angles for the cube's rotation.
_rect_rot_z = 130 * Pi / 180.0;
//+ Surface orientation: -1 for inward, 1 for outward.
_rect_surface_orient = 1;
//+ Cube model dimension: 0 for points, 1 for lines and line loops, 2 for surfaces and surface loops.
_rect_model_dim = 2;

Call CreateRect;