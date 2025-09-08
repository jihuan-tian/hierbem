## Copyright (C) 2021-2023 Jihuan Tian <jihuan_tian@hotmail.com>
##
## This file is part of the HierBEM library.
##
## HierBEM is free software: you can use it, redistribute it and/or modify it
## under the terms of the GNU Lesser General Public License as published by the
## Free Software Foundation, either version 3 of the License, or (at your
## option) any later version. The full text of the license can be found in the
## file LICENSE at the top level directory of HierBEM.

L = [1+2*i,0,0,0;2+4*i,3+6*i,0,0;4+8*i,5+10*i,6+12*i,0;7+14*i,8+16*i,9+18*i,10+20*i];
b = [3+7*i;6+4*i;9+7*i;10+5*i];
y = forward_substitution(L, b)
norm(L * y - b, 'fro') / norm(b, 'fro')
