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
load hmatrix-hmatrix-mmult-level-conserving-coarse-coarse-fine-ntp.output;

M = M1 * M2;
norm(H1_mult_H2_full - M, 'fro') / norm(M, 'fro')
norm(H3_full - M, 'fro') / norm(M, 'fro')

figure;
set_fig_size(gcf, 900, 300);
subplot(1, 3, 1);
plot_bct_struct("H1_bct.dat");
title("H1");
subplot(1, 3, 2);
plot_bct_struct("H2_bct.dat");
title("H2");
subplot(1, 3, 3);
plot_bct_struct("H3_bct.dat");
title("H3=H1*H2");

print("hmatrix-hmatrix-mmult-level-conserving-coarse-coarse-fine-ntp.png", "-dpng", "-r600");
