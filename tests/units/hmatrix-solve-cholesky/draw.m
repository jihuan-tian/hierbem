## Copyright (C) 2024 Jihuan Tian <jihuan_tian@hotmail.com>
##
## This file is part of the HierBEM library.
##
## HierBEM is free software: you can use it, redistribute it and/or modify it
## under the terms of the GNU Lesser General Public License as published by the
## Free Software Foundation, either version 3 of the License, or (at your
## option) any later version. The full text of the license can be found in the
## file LICENSE at the top level directory of HierBEM.

figure;
plot(x,'ro');
hold on;
plot(x_octave,'b+');
hold off;

figure;
set_fig_size(gcf, 900, 400);
subplot(1,2,1);
plot_bct_struct("H_bct.dat");
title("H-matrix");
subplot(1,2,2);
plot_bct_struct("L_bct.dat");
title("Cholesky factorization of H-matrix");

figure;
show_matrix(L_full);
hold on;
plot_bct_struct("L_bct.dat", false, false);
