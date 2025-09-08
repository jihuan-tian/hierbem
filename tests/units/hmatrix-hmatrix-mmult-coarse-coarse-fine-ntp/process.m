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
load hmatrix-hmatrix-mmult-coarse-coarse-fine-ntp.output;

norm(H_full - H1_mult_H2_full, 'fro') / norm(H_full, 'fro')
M = M1 * M2;
norm(M - H1_mult_H2_full, 'fro') / norm(M, 'fro')

figure;
subplot(2, 3, 1);
plot_bct_struct("H1_bct.dat");
title("H1");
subplot(2, 3, 2);
plot_bct_struct("H2_bct.dat");
title("H2");
subplot(2, 3, 3);
plot_bct_struct("H_bct.dat");
title("H=H1*H2");
subplot(2, 3, 4);
plot_bct_struct("bct3.dat");
title("Desired struct for H");
subplot(2, 3, 5);
plot_bct_struct("Tind_after_phase1.dat");
title("Tind after phase1");
subplot(2, 3, 6);
plot_bct_struct("Tind_after_phase2.dat");
title("Tind after phase2");

print("hmatrix-hmatrix-mmult-coarse-coarse-fine-ntp.png", "-dpng", "-r600");
