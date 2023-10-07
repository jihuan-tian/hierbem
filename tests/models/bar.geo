Mesh.Algorithm = 6;

Include "../../scripts/gmsh-macros/CreateCube.geo";

_cube_origin_x = 0;
_cube_origin_y = 0;
_cube_origin_z = 0;
_cube_x_dim = 1;
_cube_y_dim = 2;
_cube_z_dim = 6;
// _cube_element_size = 0.8;
_cube_element_size = 1;
_cube_rot_x = 0;
_cube_rot_y = 0;
_cube_rot_z = 0;
_cube_model_dim = 3;
Call CreateCube;

//+
Physical Volume(1) = {26};
//+
Physical Surface(1) = {19};
//+
Physical Surface(2) = {20};
//+
Physical Surface(3) = {23};
//+
Physical Surface(4) = {24};
//+
Physical Surface(5) = {21};
//+
Physical Surface(6) = {22};
