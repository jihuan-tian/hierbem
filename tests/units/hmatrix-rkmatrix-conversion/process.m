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

## Plot the partition structure of all the agglomerations.
## N = 177;

## for m = 0:N
##   figure(m + 1);
##   plot_bct_struct(cstrcat("hmat-bct", num2str(m), ".dat"), false);
##   title("Agglomeration of H-matrix");
##   number_str = sprintf("%03d", m);
##   PrintGCF(cstrcat("hmat-bct", number_str, ".png"));
## endfor

load hmatrix-rkmatrix-conversion.output;

norm(M_agglomerated.A * M_agglomerated.B' - M, "fro") / norm(M, "fro")
