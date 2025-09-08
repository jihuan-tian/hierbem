## Copyright (C) 2021-2023 Jihuan Tian <jihuan_tian@hotmail.com>
##
## This file is part of the HierBEM library.
##
## HierBEM is free software: you can use it, redistribute it and/or modify it
## under the terms of the GNU Lesser General Public License as published by the
## Free Software Foundation, either version 3 of the License, or (at your
## option) any later version. The full text of the license can be found in the
## file LICENSE at the top level directory of HierBEM.

clear all;

load_packages;
load hmatrix-coarsening.output;

norm(hmat_fine_to_full - M, 'fro') / norm(M, 'fro')
norm(hmat_coarse_to_full - M, 'fro') / norm(M, 'fro')

figure;
plot_bct_struct("bct1.dat", false);
title("Fine block cluster tree structure");
PrintGCF("bct1.png");
figure;
plot_bct_struct("bct2.dat", false);
title("Coarse block cluster tree structure");
PrintGCF("bct2.png");
figure;
plot_bct_struct("hmat_fine_partition.dat", true);
title("Fine H-matrix structure");
PrintGCF("hmat_fine_partition.png");
figure;
plot_bct_struct("hmat_coarse_partition.dat", true);
title("Coarse H-matrix structure");
PrintGCF("hmat_coarse_partition.png");
