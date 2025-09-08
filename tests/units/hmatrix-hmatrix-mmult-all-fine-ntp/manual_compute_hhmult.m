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
load hmatrix-hmatrix-mmult-all-fine-ntp.output;

M = M1 * M2;

## All matrice have cross split mode of C*C type.

## M(1:8,1:8), sigma_p={M1, M2}

## M(1:8,1:8), sigma_p = {}
## M(1:4,1:4), sigma_p = {{M1(1:4,1:4), M2(1:4,1:4)}, {M1(1:4,5:8), M2(4:8,1:4)}}
## {M1(1:4,5:8), M2(4:8,1:4)} is a rank-k matrix multiplication, which explains why the Sigma_b^R list of M(1:4,1:4) is not empty.

## M(1:4,5:8), sigma_p = {{M1(1:4,1:4), M2(1:4,4:8)}, {M1(1:4,4:8), M2(4:8,4:8)}}
## Because both M2(1:4,4:8) and M1(1:4,4:8) are rank-k matrices, the Sigma_b^R list of M(1:4,5:8) has two elements.
