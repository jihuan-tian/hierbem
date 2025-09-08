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

N = 10;

for m = 1:N
  figure(m);
  plot_bct_struct(cstrcat("bct-struct-with-rank=", num2str(m), ".dat"));
  title(cstrcat("H-matrix block cluster tree structure (rank=", num2str(m), ")"));
  PrintGCF(cstrcat("bct-struc-with-rank=", num2str(m), ".png"));
endfor
