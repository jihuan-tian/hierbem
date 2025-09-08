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
load hmatrix-fine-ntp-to-tp.output;

figure;
subplot(3, 2, 1);
plot_bct_struct("bct-fine-ntp.dat");
title("BCT1: fine non-tensor product partition");

subplot(3, 2, 2);
plot_bct_struct("bct-tp.dat");
title("BCT2: tensor product partition");

subplot(3, 2, 3);
plot_bct_struct("hmat1.dat");
title("H-matrix based on BCT1 with rank 2");

subplot(3, 2, 4);
imagesc(M_from_hmat1);
title("H-matrix data");
axis off;

subplot(3, 2, 5);
plot_bct_struct("hmat2.dat");
title("H-matrix converted to BCT2 with rank 1");

subplot(3, 2, 6);
imagesc(M_from_hmat2);
title("H-matrix data after conversion")
axis off;

print("convert_h.png", "-dpng", "-r600");

norm(M - M_from_hmat1, 'fro') / norm(M, 'fro')
norm(M - M_from_hmat2, 'fro') / norm(M, 'fro')
norm(M_from_hmat1 - M_from_hmat2, 'fro') / norm(M_from_hmat1, 'fro')
