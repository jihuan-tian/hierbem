## Copyright (C) 2023-2025 Jihuan Tian <jihuan_tian@hotmail.com>
##
## This file is part of the HierBEM library.
##
## HierBEM is free software: you can use it, redistribute it and/or modify it
## under the terms of the GNU Lesser General Public License as published by the
## Free Software Foundation, either version 3 of the License, or (at your option)
## any later version. The full text of the license can be found in the file
## LICENSE at the top level directory of HierBEM.

function plot_connecting_edge(block1, block2, arrow_length, arrow_width, arrow_type)
  drawArrow((block1(1,1) + block1(3,1)) / 2, (block1(1,2) + block1(3,2)) / 2, (block2(1,1) + block2(3,1)) / 2, (block2(1,2) + block2(3,2)) / 2, arrow_length, arrow_width, arrow_type);
endfunction
