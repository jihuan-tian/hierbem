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

Include "CreateSphere.geo";

_sphere_radius = 1;
_sphere_center_x = 0;
_sphere_center_y = 0;
_sphere_center_z = 0;
_sphere_element_size = 0.5;
_sphere_surface_orient = 1;
_sphere_model_dim = 0;
Call CreateSphere;

_sphere_radius = 1;
_sphere_center_x = 4;
_sphere_center_y = 0;
_sphere_center_z = 0;
_sphere_element_size = 0.5;
_sphere_surface_orient = 1;
_sphere_model_dim = 1;
Call CreateSphere;

_sphere_radius = 1;
_sphere_center_x = 4;
_sphere_center_y = 4;
_sphere_center_z = 0;
_sphere_element_size = 0.5;
_sphere_surface_orient = 1;
_sphere_model_dim = 2;
Call CreateSphere;

_sphere_radius = 1;
_sphere_center_x = 0;
_sphere_center_y = 4;
_sphere_center_z = 0;
_sphere_element_size = 0.5;
_sphere_surface_orient = 1;
Call CreateSphere;

_sphere_radius = 1;
_sphere_center_x = 2;
_sphere_center_y = 2;
_sphere_center_z = 2;
_sphere_element_size = 0.5;
_sphere_surface_orient = 1;
_sphere_model_dim = 3;
Call CreateSphere;

_sphere_radius = 1;
_sphere_center_x = 2;
_sphere_center_y = 2;
_sphere_center_z = -2;
_sphere_element_size = 0.5;
_sphere_surface_orient = 1;
_sphere_model_dim = 3;
Call CreateSphere;
