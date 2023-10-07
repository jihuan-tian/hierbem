Mesh.MeshSizeMax = 0.2;

//+
SetFactory("OpenCASCADE");
//+
Sphere(1) = {-1.5, 0, 0, 1, -Pi/2, Pi/2, 2*Pi};
//+
Sphere(2) = {1.5, 0, 0, 1, -Pi/2, Pi/2, 2*Pi};
