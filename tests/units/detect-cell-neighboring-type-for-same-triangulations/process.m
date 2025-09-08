## Copyright (C) 2022-2023 Jihuan Tian <jihuan_tian@hotmail.com>
##
## This file is part of the HierBEM library.
##
## HierBEM is free software: you can use it, redistribute it and/or modify it
## under the terms of the GNU Lesser General Public License as published by the
## Free Software Foundation, either version 3 of the License, or (at your
## option) any later version. The full text of the license can be found in the
## file LICENSE at the top level directory of HierBEM.

figure;
plot_support_points("case1_fe1_order=1_support_points.gpl");
PrintGCF("case1_fe1_order=1_support_points.png");

figure;
plot_support_points("case1_fe2_order=2_support_points.gpl");
PrintGCF("case1_fe2_order=2_support_points.png");

figure;
plot_support_points("case2_fe1_order=1_support_points.gpl");
PrintGCF("case2_fe1_order=1_support_points.png");

figure;
plot_support_points("case2_fe2_order=0_support_points.gpl");
PrintGCF("case2_fe2_order=0_support_points.png");

figure;
plot_support_points("case3_fe1_order=0_support_points.gpl");
PrintGCF("case3_fe1_order=0_support_points.png");

figure;
plot_support_points("case3_fe2_order=0_support_points.gpl");
PrintGCF("case3_fe2_order=0_support_points.png");
