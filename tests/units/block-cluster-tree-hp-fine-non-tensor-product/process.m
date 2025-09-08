## Copyright (C) 2022-2024 Jihuan Tian <jihuan_tian@hotmail.com>
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

figure;
subplot(3, 2, 1);
plot_bct_struct("bct1.dat", 'show_rank', false);
subplot(3, 2, 2);
plot_bct_struct("bct2.dat", 'show_rank', false);
subplot(3, 2, 3);
plot_bct_struct("bct3.dat", 'show_rank', false);
subplot(3, 2, 4);
plot_bct_struct("bct4.dat", 'show_rank', false);
subplot(3, 2, 5);
plot_bct_struct("bct5.dat", 'show_rank', false);
