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
load lapack-matrix-inverse.output;

# M = reshape([-1, 5, 2, -3, 6, 1, -2, 4, 2, -3, -4, 1, -3, -1, 1, 2, -2, 4, 2, -1, 3, 1, -1, 3, -3, 7, 2, -3, 7, 2, -2, 2, 1, 0, 0, -1, 1, -4, 0, 0, 0, 2, 0, -2, 3, -1, -1, 6, -2, 4, 3, -2, 4, -1, -1, 3, 3, -4, -6, 1, -3, -3, 1, -2], 8, 8);
norm(M_inv_from_input - inv(M), 'fro') / norm(inv(M), 'fro')
norm(M_inv - inv(M), 'fro') / norm(inv(M), 'fro')
norm(M_prime_inv - inv(M), 'fro') / norm(inv(M), 'fro')
norm(M_prime_inv - M_inv, 'fro') / norm(M_inv, 'fro')

## figure;
## set_fig_size(gcf, 1500, 800);
## center_fig;
## subplot(1, 2, 1);
## imagesc(M_inv);
## axis equal;
## axis off;
## colorbar;
## subplot(1, 2, 2);
## imagesc(inv(M));
## axis equal;
## axis off;
## colorbar;
