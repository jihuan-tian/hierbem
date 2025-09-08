## Copyright (C) 2021-2023 Jihuan Tian <jihuan_tian@hotmail.com>
##
## This file is part of the HierBEM library.
##
## HierBEM is free software: you can use it, redistribute it and/or modify it
## under the terms of the GNU Lesser General Public License as published by the
## Free Software Foundation, either version 3 of the License, or (at your
## option) any later version. The full text of the license can be found in the
## file LICENSE at the top level directory of HierBEM.

L = [1,0,0,0;2,3,0,0;4,5,6,0;7,8,9,10];
b = [3;6;9;10];
y = forward_substitution(L, b)
norm(L * y - b, 'fro') / norm(b, 'fro')
