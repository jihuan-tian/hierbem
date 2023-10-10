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
