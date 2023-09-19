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

Include "CreateCylinder.geo";

_cylinder_radius = 1;
_cylinder_height = 3;
_cylinder_bottom_center_x = 10;
_cylinder_bottom_center_y = 10;
_cylinder_bottom_center_z = -5;
_cylinder_rot_x = -5 * Pi / 180;
_cylinder_rot_y = 10 * Pi / 180;
_cylinder_rot_z = 0;
_cylinder_element_size = 1;
_cylinder_surface_orient = -1;
_cylinder_model_dim = 2;

Call CreateCylinder;

_cylinder_radius = 3;
_cylinder_height = 3;
_cylinder_bottom_center_x = -10;
_cylinder_bottom_center_y = -10;
_cylinder_bottom_center_z = 5;
_cylinder_rot_x = 15 * Pi / 180;
_cylinder_rot_y = -20 * Pi / 180;
_cylinder_rot_z = 0;
_cylinder_element_size = 1;
_cylinder_surface_orient = 1;
_cylinder_model_dim = 3;

Call CreateCylinder;
