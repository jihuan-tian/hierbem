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

Include "CreateCube.geo";

_cube_origin_x = 0;
_cube_origin_y = 0;
_cube_origin_z = 0;
_cube_x_dim = 1;
_cube_y_dim = 1;
_cube_z_dim = 1;
_cube_element_size = 0.2;
_cube_rot_x = 0;
_cube_rot_y = 0;
_cube_rot_z = 0;
_cube_model_dim = 0;
Call CreateCube;

_cube_origin_x = 4;
_cube_origin_y = 0;
_cube_origin_z = 0;
_cube_x_dim = 1;
_cube_y_dim = 1;
_cube_z_dim = 1;
_cube_element_size = 0.2;
_cube_rot_x = 30 * Pi / 180.0;
_cube_rot_y = 40 * Pi / 180.0;
_cube_rot_z = 50 * Pi / 180.0;
_cube_model_dim = 1;
Call CreateCube;

_cube_origin_x = 4;
_cube_origin_y = 4;
_cube_origin_z = 0;
_cube_x_dim = 1;
_cube_y_dim = 1;
_cube_z_dim = 1;
_cube_rot_x = 120 * Pi / 180.0;
_cube_rot_y = 100 * Pi / 180.0;
_cube_rot_z = 90 * Pi / 180.0;
_cube_element_size = 0.2;
_cube_model_dim = 2;
Call CreateCube;

_cube_origin_x = 0;
_cube_origin_y = 4;
_cube_origin_z = 0;
_cube_x_dim = 1;
_cube_y_dim = 1;
_cube_z_dim = 1;
_cube_rot_x = -50 * Pi / 180.0;
_cube_rot_y = -90 * Pi / 180.0;
_cube_rot_z = -70 * Pi / 180.0;
_cube_element_size = 0.2;
_cube_model_dim = 3;
Call CreateCube;

_cube_origin_x = 2;
_cube_origin_y = 2;
_cube_origin_z = 2;
_cube_x_dim = 1;
_cube_y_dim = 1;
_cube_z_dim = 1;
_cube_element_size = 0.2;
_cube_model_dim = 3;
Call CreateCube;

_cube_origin_x = 2;
_cube_origin_y = 2;
_cube_origin_z = -2;
_cube_x_dim = 1;
_cube_y_dim = 1;
_cube_z_dim = 1;
_cube_element_size = 0.2;
_cube_model_dim = 3;
Call CreateCube;
