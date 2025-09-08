// Copyright (C) 2023 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

Merge "spanner.stp";
// MeshSize {153,155,109,110,113,115,117,119,121,123,125,127,129,131,133,135,137,139,141,143,145,147,149,151} = 5;
// MeshSize {142,140,138,136,134,132,130,128,126,124,122,120,118,116,114,112,111,156,154,152,150,148,146,144} = 5;
// MeshSize {198,196,197,195,189,188,190,187} = 5;
// MeshSize {4,3,186,185} = 3;

// Get all surfaces
Physical Surface(0) = {1:232};
Physical Surface(0) -= {38,5};
Physical Surface(1) = {38};
Physical Surface(2) = {5};
Physical Volume(1) = {1};
