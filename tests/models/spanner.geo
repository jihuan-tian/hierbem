// Copyright (C) 2023 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

Merge "spanner.brep";

Physical Surface(0) = {1:232};
Physical Surface(0) -= {38,5};
Physical Surface(1) = {38};
Physical Surface(2) = {5};
Physical Volume(1) = {1};

Mesh.Algorithm = 6;
Mesh.SubdivisionAlgorithm = 1;
Mesh.MeshSizeMax = 20;
Mesh.MeshSizeMin = 0.05;
Mesh.MeshSizeFromCurvature = 5;
MeshSize {109:156} = 3;
MeshSize {3:4} = 3;
MeshSize {185:190} = 3;
MeshSize {195:198} = 3;