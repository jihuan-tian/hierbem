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
load "hmatrix-memory-consumption.output";

plot(hmatrix_memory_consumption(:,1), hmatrix_memory_consumption(:,2)/1024/1024, "r-o");
hold on;
plot(hmatrix_memory_consumption(:,1), hmatrix_memory_consumption(:,1).^2*8/1024/1024, "b-+");
grid on;
legend({"Compressed matrix", "Full matrix"}, "location", "southoutside");
xlabel("Matrix size");
ylabel("Memory consumption (MB)");
## title("Linear storage complexity of H-matrix for BEM");
title("Linear storage complexity achieved by compressed matrix\nfor boundary element method");
