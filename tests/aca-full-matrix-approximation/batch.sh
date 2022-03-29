#!/bin/bash

# Run calculation for SLP matrix
/home/jihuan/Projects/deal.ii/program/dealii-9.1.1/examples/laplace-bem/tests/aca-full-matrix-approximation/aca-full-matrix-approximation.release/aca-full-matrix-approximation.release -i A.dat -v A -n 4 -e 0.1 -a 1.0 > 2022-03-19-aca-plus-for-full-matrix/slp/01.dat
/home/jihuan/Projects/deal.ii/program/dealii-9.1.1/examples/laplace-bem/tests/aca-full-matrix-approximation/aca-full-matrix-approximation.release/aca-full-matrix-approximation.release -i A.dat -v A -n 4 -e 0.01 -a 1.0 > 2022-03-19-aca-plus-for-full-matrix/slp/02.dat
/home/jihuan/Projects/deal.ii/program/dealii-9.1.1/examples/laplace-bem/tests/aca-full-matrix-approximation/aca-full-matrix-approximation.release/aca-full-matrix-approximation.release -i A.dat -v A -n 8 -e 0.1 -a 1.0 > 2022-03-19-aca-plus-for-full-matrix/slp/03.dat
/home/jihuan/Projects/deal.ii/program/dealii-9.1.1/examples/laplace-bem/tests/aca-full-matrix-approximation/aca-full-matrix-approximation.release/aca-full-matrix-approximation.release -i A.dat -v A -n 8 -e 0.01 -a 1.0 > 2022-03-19-aca-plus-for-full-matrix/slp/04.dat
/home/jihuan/Projects/deal.ii/program/dealii-9.1.1/examples/laplace-bem/tests/aca-full-matrix-approximation/aca-full-matrix-approximation.release/aca-full-matrix-approximation.release -i A.dat -v A -n 16 -e 0.1 -a 1.0 > 2022-03-19-aca-plus-for-full-matrix/slp/05.dat
/home/jihuan/Projects/deal.ii/program/dealii-9.1.1/examples/laplace-bem/tests/aca-full-matrix-approximation/aca-full-matrix-approximation.release/aca-full-matrix-approximation.release -i A.dat -v A -n 16 -e 0.01 -a 1.0 > 2022-03-19-aca-plus-for-full-matrix/slp/06.dat

# Run calculation for DLP matrix
/home/jihuan/Projects/deal.ii/program/dealii-9.1.1/examples/laplace-bem/tests/aca-full-matrix-approximation/aca-full-matrix-approximation.release/aca-full-matrix-approximation.release -i B.dat -v A -n 4 -e 0.1 -a 1.0 > 2022-03-19-aca-plus-for-full-matrix/dlp/01.dat
/home/jihuan/Projects/deal.ii/program/dealii-9.1.1/examples/laplace-bem/tests/aca-full-matrix-approximation/aca-full-matrix-approximation.release/aca-full-matrix-approximation.release -i B.dat -v A -n 4 -e 0.01 -a 1.0 > 2022-03-19-aca-plus-for-full-matrix/dlp/02.dat
/home/jihuan/Projects/deal.ii/program/dealii-9.1.1/examples/laplace-bem/tests/aca-full-matrix-approximation/aca-full-matrix-approximation.release/aca-full-matrix-approximation.release -i B.dat -v A -n 8 -e 0.1 -a 1.0 > 2022-03-19-aca-plus-for-full-matrix/dlp/03.dat
/home/jihuan/Projects/deal.ii/program/dealii-9.1.1/examples/laplace-bem/tests/aca-full-matrix-approximation/aca-full-matrix-approximation.release/aca-full-matrix-approximation.release -i B.dat -v A -n 8 -e 0.01 -a 1.0 > 2022-03-19-aca-plus-for-full-matrix/dlp/04.dat
/home/jihuan/Projects/deal.ii/program/dealii-9.1.1/examples/laplace-bem/tests/aca-full-matrix-approximation/aca-full-matrix-approximation.release/aca-full-matrix-approximation.release -i B.dat -v A -n 16 -e 0.1 -a 1.0 > 2022-03-19-aca-plus-for-full-matrix/dlp/05.dat
/home/jihuan/Projects/deal.ii/program/dealii-9.1.1/examples/laplace-bem/tests/aca-full-matrix-approximation/aca-full-matrix-approximation.release/aca-full-matrix-approximation.release -i B.dat -v A -n 16 -e 0.01 -a 1.0 > 2022-03-19-aca-plus-for-full-matrix/dlp/06.dat
