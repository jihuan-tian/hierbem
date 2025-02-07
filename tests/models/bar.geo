SetFactory("OpenCASCADE");
Merge "bar.brep";
Mesh.Algorithm = 11;
Mesh.SubdivisionAlgorithm = 0;
Mesh.MeshSizeExtendFromBoundary = 0;
Mesh.MeshSizeFromCurvature = 0;
Mesh.MeshSizeFromPoints = 0;

Field[1] = MathEval;
Field[1].F = "0.5";
Background Field = 1;