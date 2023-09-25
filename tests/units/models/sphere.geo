Mesh.Algorithm = 6;

Include "../../scripts/gmsh-macros/CreateSphere.geo";

_sphere_radius = 1;
_sphere_center_x = 0;
_sphere_center_y = 0;
_sphere_center_z = 0;
_sphere_surface_orient = -1;
_sphere_element_size = 0.8;
//+ Sphere model dimension: 0 for points, 1 for lines and line loops, 2 for surfaces and surface loops, 3 for volume.
_sphere_model_dim = 3;

Call CreateSphere;
