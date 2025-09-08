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
load hmatrix-invert-by-gauss-elim.output;

M_inv = inv(M);
H_before_inv_full_inv = inv(H_before_inv_full);

norm(H_before_inv_full_inv - H_inv_full, 'fro') / norm(H_before_inv_full_inv, 'fro')
norm(M_inv - H_inv_full, 'fro') / norm(M_inv, 'fro')

figure;
set_fig_size(gcf, 900, 300);
subplot(1, 3, 1);
plot_bct_struct("H_before_inv_bct.dat");
title("H before inverse");

subplot(1, 3, 2);
plot_bct_struct("H_after_inv_bct.dat");
title("H after inverse");

subplot(1, 3, 3);
plot_bct_struct("H_inv_bct.dat");
title("Inverse of H");

print("hmatrix-invert-by-gauss-elim.png", "-dpng", "-r600");
