#!/bin/bash

for rank in {1..8}; do
    echo "=== Truncation rank=$rank, executing..."
    ./hmatrix-solve-cholesky-in-situ-for-erichsen1996efficient-example2.debug/hmatrix-solve-cholesky-in-situ-for-erichsen1996efficient-example2.debug build_hmat $rank spd > slp-spd-truncation-rank=$rank.dat
done
