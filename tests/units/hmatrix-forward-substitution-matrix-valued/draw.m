## Copyright (C) 2023 Jihuan Tian <jihuan_tian@hotmail.com>
##
## This file is part of the HierBEM library.
##
## HierBEM is free software: you can use it, redistribute it and/or modify it
## under the terms of the GNU Lesser General Public License as published by the
## Free Software Foundation, either version 3 of the License, or (at your
## option) any later version. The full text of the license can be found in the
## file LICENSE at the top level directory of HierBEM.

load_packages;

## Plot the error of the solution matrix.
figure;
show_matrix(X - X_octave);

## Plot the block cluster tree structures.
figure;
set_fig_size(gcf, [900, 300]);
subplot(1, 3, 1);
plot_bct_struct("HL_bct.dat");
title("Lower triangular matrix");
subplot(1, 3, 2);
plot_bct_struct("HZ_bct.dat");
title("RHS matrix");
subplot(1, 3, 3);
plot_bct_struct("HX_bct.dat");
title("Solution matrix");
